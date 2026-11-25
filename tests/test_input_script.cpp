// Scripted controller input for the native boot harness (runtime/guest/input_script.h).
#include <chrono>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

#include "abi/playgta6_abi.h"
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

using playgta6::guest::InputScriptEvent;
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

}  // namespace

int main() {
    test_parse();
    test_player_presses_and_releases();
    test_title_reads_the_script_through_game_input();
    if (g_failures != 0) {
        std::fprintf(stderr, "test_input_script: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test_input_script: all checks passed\n");
    return 0;
}
