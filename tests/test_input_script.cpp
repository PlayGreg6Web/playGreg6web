// Scripted controller input for the native boot harness (runtime/guest/input_script.h).
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>
#include <vector>

#include "abi/playgta6_abi.h"
#include "guest/input_record.h"
#include "guest/input_script.h"
#include "os/game_input.h"
#include "os/xbox_scarlett.h"

namespace {

int g_failures = 0;

#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::fprintf(stderr, "FAIL [%s:%d]: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                    \
        }                                                                    \
    } while (0)

using playgta6::guest::ConditionTracker;
using playgta6::guest::InputCondition;
using playgta6::guest::InputScriptEvent;
using playgta6::guest::InputStatus;
using playgta6::guest::InputStepKind;
using playgta6::guest::parse_input_script;
namespace button = playgta6::os::input_button;

void test_parse() {
    std::vector<InputScriptEvent> ev;
    std::string error;
    CHECK(parse_input_script("# press A, then Menu+B\n"
                             "\n"
                             "  3000 Menu+B 250   # comment\n"
                             "1000 A\n"
                             "2000 0x300 50\n",
                             ev, error));
    CHECK(ev.size() == 3);
    if (ev.size() == 3) {
        // Sorted by time; hold defaults to 100 ms.
        CHECK(ev[0].time_ms == 1000 && ev[0].buttons == button::kGamepadA && ev[0].hold_ms == 100);
        CHECK(ev[1].time_ms == 2000 && ev[1].buttons == 0x300 && ev[1].hold_ms == 50);
        CHECK(ev[2].time_ms == 3000 && ev[2].buttons == (button::kMenu | button::kGamepadB) && ev[2].hold_ms == 250);
    }
    CHECK(parse_input_script("", ev, error) && ev.empty());
    // Stick directions for menu navigation.
    CHECK(parse_input_script("10 Down\n20 Up+A\n30 Left\n40 Right\n", ev, error) && ev.size() == 4);
    if (ev.size() == 4) {
        CHECK(ev[0].buttons == button::kMoveBackward);
        CHECK(ev[1].buttons == (button::kMoveForward | button::kGamepadA));
        CHECK(ev[2].buttons == button::kMoveLeft && ev[3].buttons == button::kMoveRight);
    }
    // D-pad and stick clicks.
    CHECK(parse_input_script("10 DDown\n20 DUp+A\n30 DLeft\n40 DRight\n50 LS+RS\n", ev, error) && ev.size() == 5);
    if (ev.size() == 5) {
        CHECK(ev[0].buttons == button::kDPadDown);
        CHECK(ev[1].buttons == (button::kDPadUp | button::kGamepadA));
        CHECK(ev[2].buttons == button::kDPadLeft && ev[3].buttons == button::kDPadRight);
        CHECK(ev[4].buttons == (button::kLeftStickPress | button::kRightStickPress));
    }
    // Right stick and triggers.
    CHECK(parse_input_script("10 RUp\n20 RDown+A\n30 RLeft+RT\n40 RRight+LT\n50 LT+RT\n", ev, error) &&
          ev.size() == 5);
    if (ev.size() == 5) {
        CHECK(ev[0].right_stick_y == 1.0f && ev[0].buttons == 0);
        CHECK(ev[1].right_stick_y == -1.0f && ev[1].buttons == button::kGamepadA);
        CHECK(ev[2].right_stick_x == -1.0f && ev[2].right_trigger == 1.0f && ev[2].left_trigger == 0.0f);
        CHECK(ev[3].right_stick_x == 1.0f && ev[3].left_trigger == 1.0f);
        CHECK(ev[4].left_trigger == 1.0f && ev[4].right_trigger == 1.0f && ev[4].right_stick_x == 0.0f);
    }
    CHECK(!parse_input_script("10 RUpp\n", ev, error));

    // Malformed lines are refused with their line number.
    CHECK(!parse_input_script("1000 A\nsoon A\n", ev, error) && error.rfind("line 2:", 0) == 0);
    CHECK(!parse_input_script("1000 Start\n", ev, error) && error.find("bad buttons") != std::string::npos);
    CHECK(!parse_input_script("1000\n", ev, error) && error.find("bad buttons") != std::string::npos);
    CHECK(!parse_input_script("1000 A+\n", ev, error));
    CHECK(!parse_input_script("1000 0\n", ev, error));
    CHECK(!parse_input_script("1000 A 0\n", ev, error) && error.find("bad hold") != std::string::npos);
    CHECK(!parse_input_script("1000 A 100 extra\n", ev, error) && error.find("unexpected") != std::string::npos);
}

void test_player_presses_and_releases() {
    playgta6::InputState state{};
    playgta6::os::GameInputReader reader(&state);
    {
        playgta6::guest::InputScriptPlayer player({{0, button::kGamepadA, 200}}, &state);
        player.start();
        // The press lands at once and is held for 200 ms.
        bool seen_down = false;
        for (int i = 0; i < 100 && !seen_down; ++i) {
            seen_down = reader.buttons() == button::kGamepadA;
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        CHECK(seen_down);
        bool released = false;
        for (int i = 0; i < 200 && !released; ++i) {
            released = player.writes() == 2 && reader.buttons() == 0;
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        CHECK(released);
        CHECK(state.seq == 2);  // one write per press and per release
    }

    // Right stick and triggers reach the gamepad state, then clear on release.
    {
        playgta6::InputState analog{};
        playgta6::os::GameInputReader areader(&analog);
        std::vector<InputScriptEvent> ev;
        std::string error;
        CHECK(parse_input_script("0 RUp+RRight+LT+RT 200\n", ev, error) && ev.size() == 1);
        playgta6::guest::InputScriptPlayer player(ev, &analog);
        player.start();
        playgta6::os::GameInputFrame frame;
        bool held = false;
        for (int i = 0; i < 100 && !held; ++i) {
            if (areader.poll(frame, false) && frame.left_trigger > 0.9f) {
                held = frame.right_stick.y > 0.5f && frame.right_stick.x > 0.5f && frame.right_trigger > 0.9f;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        CHECK(held);
        bool released = false;
        for (int i = 0; i < 200 && !released; ++i) {
            released = player.writes() == 2 && areader.poll(frame, false) && frame.right_stick.y == 0.0f &&
                       frame.right_stick.x == 0.0f && frame.left_trigger == 0.0f && frame.right_trigger == 0.0f;
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        CHECK(released);
    }

    // stop() ends playback before a later event and leaves nothing pressed.
    playgta6::InputState idle{};
    playgta6::guest::InputScriptPlayer later({{60000, button::kGamepadB, 100}}, &idle);
    later.start();
    later.stop();
    CHECK(later.writes() == 0 && idle.buttons == 0);
}

void test_parse_screen_steps() {
    std::vector<InputScriptEvent> ev;
    std::string error;
    CHECK(parse_input_script("2000 A\n"
                             "1000 B\n"
                             "wait draws_per_frame==33|50 for 2s timeout 1.5s\n"
                             "500 DUp 150\n"
                             "0 A\n"
                             "repeat every 14s until draws==33 for 500ms timeout 400s\n"
                             "  6500 A\n"
                             "  0 DUp\n"
                             "end\n"
                             "capture\n"
                             "wait presents>=1200\n"
                             "100 A\n",
                             ev, error));
    CHECK(ev.size() == 12);
    if (ev.size() == 12) {
        // Presses are sorted only within their run.
        CHECK(ev[0].time_ms == 1000 && ev[1].time_ms == 2000);
        CHECK(ev[2].kind == InputStepKind::Wait && ev[2].for_ms == 2000 && ev[2].timeout_ms == 1500);
        CHECK(ev[2].cond.metric == InputCondition::Metric::DrawsPerFrame && ev[2].cond.op == InputCondition::Op::Eq);
        CHECK(ev[2].cond.values == std::vector<uint64_t>({33, 50}));
        CHECK(ev[3].time_ms == 0 && ev[4].time_ms == 500 && ev[4].buttons == button::kDPadUp && ev[4].hold_ms == 150);
        CHECK(ev[5].kind == InputStepKind::RepeatBegin && ev[5].every_ms == 14000 && ev[5].for_ms == 500 &&
              ev[5].timeout_ms == 400000 && ev[5].end_index == 8);
        CHECK(ev[6].time_ms == 0 && ev[7].time_ms == 6500);
        CHECK(ev[8].kind == InputStepKind::RepeatEnd && ev[9].kind == InputStepKind::Capture);
        CHECK(ev[10].cond.metric == InputCondition::Metric::Presents && ev[10].cond.op == InputCondition::Op::Ge &&
              ev[10].cond.values[0] == 1200 && ev[10].for_ms == 0 && ev[10].timeout_ms == 0);
        CHECK(playgta6::guest::condition_text(ev[2].cond) == "draws_per_frame==33|50");
    }
    // Operators, durations.
    CHECK(parse_input_script("wait draws!=4|0 for 250\nwait presents<10\nwait presents>3 timeout 2s\n", ev, error) &&
          ev.size() == 3 && ev[0].cond.op == InputCondition::Op::Ne && ev[0].for_ms == 250 &&
          ev[1].cond.op == InputCondition::Op::Lt && ev[2].cond.op == InputCondition::Op::Gt);
    // Malformed steps.
    CHECK(!parse_input_script("wait fps==30\n", ev, error) && error.find("bad condition") != std::string::npos);
    CHECK(!parse_input_script("wait draws>=3|4\n", ev, error));  // alternatives only for == / !=
    CHECK(!parse_input_script("wait draws==\n", ev, error));
    CHECK(!parse_input_script("wait draws==3 for\n", ev, error) && error.find("bad duration") != std::string::npos);
    CHECK(!parse_input_script("wait draws==3 for 2m\n", ev, error));
    CHECK(!parse_input_script("wait draws==3 soon 2s\n", ev, error) && error.find("unexpected") != std::string::npos);
    CHECK(!parse_input_script("repeat until draws==3\n0 A\nend\n", ev, error));
    CHECK(!parse_input_script("repeat every 0s until draws==3\nend\n", ev, error));
    CHECK(!parse_input_script("repeat every 1s until draws==3\n0 A\n", ev, error) && error.find("without 'end'") != std::string::npos);
    CHECK(!parse_input_script("end\n", ev, error) && error.find("without 'repeat'") != std::string::npos);
    CHECK(!parse_input_script("repeat every 1s until draws==3\nrepeat every 1s until draws==4\nend\nend\n", ev, error));
    CHECK(!parse_input_script("capture now\n", ev, error));
}

void test_condition_tracker() {
    InputCondition c;
    c.op = InputCondition::Op::Eq;
    c.values = {33, 50};
    CHECK(playgta6::guest::condition_holds(c, InputStatus{0, 33}));
    CHECK(!playgta6::guest::condition_holds(c, InputStatus{0, 4}));
    // Held continuously for 2 s; a blip restarts the clock.
    ConditionTracker t(c, 2000);
    CHECK(!t.update(0, InputStatus{10, 4}));
    CHECK(!t.update(100, InputStatus{11, 33}) && t.holds_now());
    CHECK(!t.update(1500, InputStatus{20, 50}));
    CHECK(!t.update(1600, InputStatus{21, 3}) && !t.holds_now());
    CHECK(!t.update(1700, InputStatus{22, 33}));
    CHECK(!t.update(3600, InputStatus{40, 33}));
    CHECK(t.update(3700, InputStatus{41, 33}) && t.met());
    CHECK(t.update(3800, InputStatus{42, 4}));  // stays met
    // for 0: met at the first sample that holds.
    InputCondition p;
    p.metric = InputCondition::Metric::Presents;
    p.op = InputCondition::Op::Ge;
    p.values = {100};
    ConditionTracker now(p, 0);
    CHECK(!now.update(5, InputStatus{99, 0}) && now.update(6, InputStatus{100, 0}));
}

void test_player_waits_for_the_screen() {
    // A fake present counter: the "screen" switches to 33 draws once `screen` is set.
    static std::atomic<uint64_t> draws{4};
    static std::atomic<uint64_t> presents{0};
    std::vector<InputScriptEvent> ev;
    std::string error;
    CHECK(parse_input_script("wait draws_per_frame==33 for 50ms\n"
                             "0 A 20\n"
                             "capture\n"
                             "wait presents>=1000000 timeout 30ms\n"
                             "0 B 20\n",
                             ev, error));
    playgta6::InputState state{};
    playgta6::os::GameInputReader reader(&state);
    std::atomic<int> captured{0};
    playgta6::guest::InputScriptPlayer player(ev, &state);
    player.set_status_probe([] { return InputStatus{presents.fetch_add(1), draws.load()}; });
    player.set_on_capture([&] { captured.fetch_add(1); });
    player.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    CHECK(player.writes() == 0 && state.seq == 0);  // still on the 4-draw screen: nothing pressed
    draws.store(33);
    bool done = false;
    for (int i = 0; i < 400 && !done; ++i) {
        done = player.finished();
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    CHECK(done);
    CHECK(player.writes() == 4);  // A down/up, B down/up
    CHECK(captured.load() == 1);
    CHECK(player.waits_done() == 2 && player.waits_timed_out() == 1);
    CHECK(reader.buttons() == 0);
}

void test_player_repeat_stops_on_the_target_screen() {
    // The body presses A every round until the status reads 33 draws; the screen flips after the
    // second press, so exactly two presses go out and the repeat ends met.
    static std::atomic<uint64_t> draws{20};
    draws.store(20);
    std::vector<InputScriptEvent> ev;
    std::string error;
    CHECK(parse_input_script("repeat every 200ms until draws==33 for 30ms timeout 5s\n"
                             "0 A 10\n"
                             "end\n",
                             ev, error));
    playgta6::InputState state{};
    playgta6::guest::InputScriptPlayer player(ev, &state);
    player.set_status_probe([&player] {
        if (player.writes() >= 4) {
            draws.store(33);  // the second press (writes 3 and 4) changed the screen
        }
        return InputStatus{0, draws.load()};
    });
    player.start();
    bool done = false;
    for (int i = 0; i < 400 && !done; ++i) {
        done = player.finished();
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    CHECK(done);
    CHECK(player.writes() == 4);  // two presses, two releases
    CHECK(player.waits_done() == 1 && player.waits_timed_out() == 0);
}

void test_title_reads_the_script_through_game_input() {
    // The harness attaches the script's InputState the way the browser bridge attaches its buffer.
    playgta6::InputState state{};
    playgta6::os::scarlett::set_game_input_state(&state);
    playgta6::InputState values{};
    values.buttons = button::kGamepadA;
    playgta6::os::GameInputReader::write_monotonic(state, values);
    CHECK(playgta6::os::scarlett::get_game_input_state() == &state);
    playgta6::os::GameInputReader reader(playgta6::os::scarlett::get_game_input_state());
    CHECK(reader.buttons() == button::kGamepadA);
    playgta6::os::scarlett::set_game_input_state(nullptr);
}


// record -> replay: the recorded host readings come back identical, in order, at their present index.
void test_record_replay_round_trip() {
    namespace fs = std::filesystem;
    const std::string path = (fs::temp_directory_path() / "playgta6_input_record_test.txt").string();
    std::atomic<uint64_t> presents{0};
    std::vector<std::pair<uint64_t, playgta6::InputState>> truth;
    {
        playgta6::guest::InputRecorder rec;
        std::string error;
        CHECK(rec.open(path, [&] { return presents.load(); }, error));
        auto feed = [&](uint64_t at, playgta6::InputState s) {
            presents = at;
            rec.record(s);
            rec.record(s);  // an unchanged reading is not recorded again
            truth.emplace_back(at, s);
        };
        playgta6::InputState a{};
        a.buttons = button::kGamepadA;
        feed(5, a);
        a.left_stick_x = 0.123456789f;
        a.left_stick_y = -0.75f;
        a.right_trigger = 1.0f;
        feed(5, a);  // two changes within one present both survive
        playgta6::InputState b{};
        b.buttons = button::kMenu | button::kDPadUp;
        b.right_stick_x = -1.0f;
        feed(40, b);
        feed(41, playgta6::InputState{});
        CHECK(rec.lines() == 4);
    }
    std::ifstream in(path, std::ios::binary);
    const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    CHECK(text.rfind(playgta6::guest::kInputScriptVersionLine, 0) == 0);
    std::vector<InputScriptEvent> ev;
    std::string error;
    CHECK(parse_input_script(text, ev, error));
    CHECK(ev.size() == truth.size());
    for (size_t i = 0; i < ev.size() && i < truth.size(); ++i) {
        const playgta6::InputState& s = truth[i].second;
        CHECK(ev[i].kind == InputStepKind::State && ev[i].present == truth[i].first);
        CHECK(ev[i].buttons == s.buttons && ev[i].left_stick_x == s.left_stick_x && ev[i].left_stick_y == s.left_stick_y &&
              ev[i].right_stick_x == s.right_stick_x && ev[i].right_stick_y == s.right_stick_y &&
              ev[i].left_trigger == s.left_trigger && ev[i].right_trigger == s.right_trigger);
    }
    CHECK(!parse_input_script("state 3 0x1 0 0 0 0\n", ev, error));  // too few fields
    CHECK(!parse_input_script("state 3 zz 0 0 0 0 0 0\n", ev, error));

    // Replay: the player writes each state once its present index is reached, nothing earlier.
    parse_input_script(text, ev, error);
    playgta6::InputState out{};
    std::atomic<uint64_t> live{0};
    playgta6::guest::InputScriptPlayer player(ev, &out);
    player.set_status_probe([&] { return InputStatus{live.load(), 0}; });
    player.start();
    auto wait_writes = [&](uint64_t n) {
        for (int i = 0; i < 2000 && player.writes() < n; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        return player.writes() == n;
    };
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    CHECK(player.writes() == 0);
    live = 4;
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    CHECK(player.writes() == 0);
    live = 5;
    CHECK(wait_writes(2));
    CHECK(out.buttons == truth[1].second.buttons && out.left_stick_x == truth[1].second.left_stick_x &&
          out.right_trigger == 1.0f);
    live = 40;
    CHECK(wait_writes(3));
    CHECK(out.buttons == truth[2].second.buttons && out.right_stick_x == -1.0f && out.left_stick_x == 0.0f);
    live = 41;
    CHECK(wait_writes(4));
    CHECK(out.buttons == 0 && out.right_stick_x == 0.0f);
    player.stop();
    std::error_code ec;
    fs::remove(path, ec);
}

}  // namespace

int main() {
    test_parse();
    test_player_presses_and_releases();
    test_parse_screen_steps();
    test_condition_tracker();
    test_player_waits_for_the_screen();
    test_player_repeat_stops_on_the_target_screen();
    test_title_reads_the_script_through_game_input();
    test_record_replay_round_trip();
    if (g_failures != 0) {
        std::fprintf(stderr, "test_input_script: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test_input_script: all checks passed\n");
    return 0;
}
