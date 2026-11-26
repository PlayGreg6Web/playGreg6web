// keymap.ini parsing and key/mouse -> pad mapping (runtime/guest/keymap.h).
#include <cmath>
#include <cstdio>
#include <set>
#include <string>
#include <vector>

#include "guest/host_input.h"
#include "guest/keymap.h"
#include "os/game_input.h"

namespace {
int g_failures = 0;
#define CHECK(cond)                                                                \
    do {                                                                           \
        if (!(cond)) {                                                             \
            std::fprintf(stderr, "FAIL [%s:%d]: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                          \
        }                                                                          \
    } while (0)
bool near(float a, float b, float eps = 1e-4f) { return std::fabs(a - b) < eps; }

namespace km = playgta6::guest::keymap;
namespace hi = playgta6::guest::host_input;
namespace button = playgta6::os::input_button;

playgta6::InputState eval(const km::Keymap& m, std::set<int> down, float dx = 0, float dy = 0) {
    return km::evaluate(m, [&](int vk) { return down.count(vk) != 0; }, dx, dy);
}
bool parse(const std::string& text, km::Keymap& m, std::vector<std::string>& errors) {
    return km::parse_keymap(text, m, errors);
}
bool has_error(const std::vector<std::string>& errors, const std::string& needle) {
    for (const std::string& e : errors) {
        if (e.find(needle) != std::string::npos) return true;
    }
    return false;
}
}  // namespace

int main() {
    // The default table equals the built-in keyboard conversion, key by key.
    {
        const km::Keymap d = km::default_keymap();
        struct P { int vk; uint32_t key; };
        const P pairs[] = {
            {'W', hi::kKeyW}, {'A', hi::kKeyA}, {'S', hi::kKeyS}, {'D', hi::kKeyD}, {'I', hi::kKeyI}, {'J', hi::kKeyJ},
            {'K', hi::kKeyK}, {'L', hi::kKeyL}, {0x26, hi::kKeyUp}, {0x28, hi::kKeyDown}, {0x25, hi::kKeyLeft},
            {0x27, hi::kKeyRight}, {0x20, hi::kKeySpace}, {0x0D, hi::kKeyEnter}, {0x08, hi::kKeyBackspace},
            {'F', hi::kKeyF}, {'R', hi::kKeyR}, {'Q', hi::kKeyQ}, {'E', hi::kKeyE}, {0x1B, hi::kKeyEscape},
            {0x09, hi::kKeyTab}, {0x10, hi::kKeyShift}, {0x11, hi::kKeyCtrl}, {'Z', hi::kKeyZ}, {'C', hi::kKeyC},
        };
        for (const P& p : pairs) {
            const playgta6::InputState a = eval(d, {p.vk});
            const playgta6::InputState b = hi::convert_keyboard(p.key);
            CHECK(a.buttons == b.buttons);
            CHECK(near(a.left_stick_x, b.left_stick_x) && near(a.left_stick_y, b.left_stick_y));
            CHECK(near(a.right_stick_x, b.right_stick_x) && near(a.right_stick_y, b.right_stick_y));
            CHECK(near(a.left_trigger, b.left_trigger) && near(a.right_trigger, b.right_trigger));
        }
        // Opposite keys cancel.
        CHECK(near(eval(d, {'A', 'D'}).left_stick_x, 0.0f));
        CHECK(!d.mouse_look);
    }

    // The documented default file parses back to the default table.
    {
        km::Keymap m;
        std::vector<std::string> e;
        CHECK(parse(km::default_keymap_text(), m, e));
        CHECK(e.empty());
        const km::Keymap d = km::default_keymap();
        CHECK(m.keys.size() == d.keys.size());
        CHECK(!m.mouse_look && near(m.mouse_sensitivity, 0.05f) && !m.mouse_invert_y);
    }

    // Rebinding: button, trigger, stick, unbind; comments and case-insensitivity.
    {
        km::Keymap m;
        std::vector<std::string> e;
        const std::string text =
            "# my keys\n[Keys]\nX = right_trigger   ; fire\nenter = b\nSpace = none\nMouse_Left = a\nF5 = menu\n"
            "Numpad8 = left_stick_up\n";
        CHECK(parse(text, m, e));
        CHECK(near(eval(m, {'X'}).right_trigger, 1.0f));
        CHECK(eval(m, {0x0D}).buttons == button::kGamepadB);
        CHECK(eval(m, {0x20}).buttons == 0);
        CHECK(eval(m, {0x01}).buttons == button::kGamepadA);
        CHECK(eval(m, {0x74}).buttons == button::kMenu);
        CHECK(near(eval(m, {0x68}).left_stick_y, 1.0f));
        CHECK(eval(m, {'W'}).left_stick_y > 0.5f);  // untouched defaults remain
    }

    // Two keys into the same stick direction stay clamped; a key moved off an axis frees it.
    {
        km::Keymap m;
        std::vector<std::string> e;
        CHECK(parse("[keys]\nUp = left_stick_up\nW = none\n", m, e));
        CHECK(near(eval(m, {0x26}).left_stick_y, 1.0f));
        CHECK(eval(m, {0x26}).buttons == 0);
        CHECK(near(eval(m, {'W'}).left_stick_y, 0.0f));
        CHECK(near(eval(m, {0x26, 'W'}).left_stick_y, 1.0f));
    }

    // Mouse look: right stick, sensitivity, invert, clamping, off by default.
    {
        km::Keymap m;
        std::vector<std::string> e;
        CHECK(parse("[mouse]\nlook = on\nsensitivity = 0.1\n", m, e));
        CHECK(m.mouse_look);
        playgta6::InputState s = eval(m, {}, 4.0f, 3.0f);
        CHECK(near(s.right_stick_x, 0.4f));
        CHECK(near(s.right_stick_y, -0.3f));  // mouse down = stick down
        CHECK(near(eval(m, {}, 500.0f, -500.0f).right_stick_x, 1.0f));
        CHECK(near(eval(m, {}, 500.0f, -500.0f).right_stick_y, 1.0f));
        CHECK(parse("[mouse]\nlook = on\ninvert_y = yes\n", m, e));
        CHECK(near(eval(m, {}, 0.0f, 10.0f).right_stick_y, 0.5f));
        CHECK(near(eval(m, {'L'}, -10.0f, 0.0f).right_stick_x, 0.5f));  // key +1, mouse -0.5 at 0.05
        CHECK(parse("[mouse]\nsensitivity = 0.2\n", m, e));
        CHECK(near(eval(m, {}, 5.0f, 5.0f).right_stick_x, 0.0f));  // look still off
    }

    // Errors name the line and the problem; all problems are reported.
    {
        km::Keymap m;
        std::vector<std::string> e;
        CHECK(!parse("[keys]\nFoo = a\nW = bogus\nEnter a\n[mouse]\nsensitivity = fast\nlook = maybe\nspeed = 1\n[other]\n", m, e));
        CHECK(has_error(e, "line 2: unknown key 'Foo'"));
        CHECK(has_error(e, "line 3: unknown target 'bogus'"));
        CHECK(has_error(e, "line 4: expected 'name = value'"));
        CHECK(has_error(e, "line 6: sensitivity must be a number"));
        CHECK(has_error(e, "line 7: look must be on or off"));
        CHECK(has_error(e, "line 8: unknown mouse setting"));
        CHECK(has_error(e, "line 9: unknown section"));
        CHECK(e.size() == 7);

        CHECK(!parse("[keys]\nQ = a\nq = b\n", m, e));
        CHECK(has_error(e, "line 3: key 'q' is already bound on line 2"));
        CHECK(!parse("W = a\n", m, e));
        CHECK(has_error(e, "outside a [keys] or [mouse] section"));
        CHECK(!parse("[mouse]\nsensitivity = 0\n", m, e));
        CHECK(!parse("[mouse]\nsensitivity = 11\n", m, e));
        CHECK(!parse("[keys]\nF13 = a\n", m, e));
        CHECK(parse("", m, e) && e.empty());
        CHECK(parse("\r\n[keys]\r\nW = a\r\n", m, e));  // CRLF files
        CHECK(eval(m, {'W'}).buttons == button::kGamepadA);
    }

    if (g_failures == 0) std::printf("test_keymap: all tests passed\n");
    return g_failures == 0 ? 0 : 1;
}
