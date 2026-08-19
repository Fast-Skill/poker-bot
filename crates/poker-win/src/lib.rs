//! Windows platform layer: find a window, size it, capture it, click it.
//!
//! Everything the bot needs from the operating system lives here, and nothing
//! else in the workspace uses `unsafe`. Keeping the FFI in one small, tested
//! crate means a mistake in a pointer or a struct layout has one place to hide.
//!
//! # Why sizing comes first
//!
//! The client's window is resizable, and card templates are pixel-exact —
//! measured at a 1430x1040 table, matching to within a few grey levels. At a
//! different size the layout reflows rather than scaling, so scaled templates
//! would not recover it. The bot therefore *sets* the window to a known size
//! before reading anything.
//!
//! # Capture
//!
//! Frames come back as raw RGB, the format [`poker_vision::Frame`] expects. No
//! image file is involved: PNG was only ever the format for offline training
//! captures.

#![cfg(windows)]

use std::ffi::c_void;
use std::fmt;

type Handle = *mut c_void;
type Bool = i32;

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct Rect {
    left: i32,
    top: i32,
    right: i32,
    bottom: i32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct Point {
    x: i32,
    y: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MouseInput {
    dx: i32,
    dy: i32,
    mouse_data: u32,
    flags: u32,
    time: u32,
    extra: usize,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct KeyboardInput {
    virtual_key: u16,
    scan_code: u16,
    flags: u32,
    time: u32,
    extra: usize,
}

/// The union `INPUT` carries, modelled as one rather than approximated.
///
/// An earlier version declared only the mouse member and reached the keyboard
/// one by transmuting into it. The compiler rejected that once a keyboard
/// member existed, for the good reason that the two are different sizes — and
/// writing the smaller through the larger would have left the tail
/// uninitialised. Declaring the union lets the compiler size and align it.
#[repr(C)]
#[derive(Clone, Copy)]
union InputValue {
    mouse: MouseInput,
    keyboard: KeyboardInput,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Input {
    kind: u32,
    value: InputValue,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct BitmapInfoHeader {
    size: u32,
    width: i32,
    height: i32,
    planes: u16,
    bit_count: u16,
    compression: u32,
    size_image: u32,
    x_pels_per_meter: i32,
    y_pels_per_meter: i32,
    clr_used: u32,
    clr_important: u32,
}

#[repr(C)]
struct BitmapInfo {
    header: BitmapInfoHeader,
    colors: [u32; 3],
}

const INPUT_MOUSE: u32 = 0;
const INPUT_KEYBOARD: u32 = 1;
/// Send the character itself rather than a key code, so the layout the user
/// happens to have does not change what arrives.
const KEYEVENTF_UNICODE: u32 = 0x0004;
const KEYEVENTF_KEYUP: u32 = 0x0002;
const VK_BACK: u16 = 0x08;
const VK_CONTROL: u16 = 0x11;
const VK_A: u16 = 0x41;
const MOUSEEVENTF_LEFTDOWN: u32 = 0x0002;
const MOUSEEVENTF_LEFTUP: u32 = 0x0004;
const SWP_NOMOVE: u32 = 0x0002;
const SWP_NOZORDER: u32 = 0x0004;
const SRCCOPY: u32 = 0x00CC_0020;
const DIB_RGB_COLORS: u32 = 0;
const BI_RGB: u32 = 0;

#[link(name = "user32")]
extern "system" {
    fn EnumWindows(callback: extern "system" fn(Handle, isize) -> Bool, param: isize) -> Bool;
    fn IsWindowVisible(window: Handle) -> Bool;
    fn GetWindowRect(window: Handle, rect: *mut Rect) -> Bool;
    fn GetWindowTextW(window: Handle, text: *mut u16, len: i32) -> i32;
    fn GetWindowThreadProcessId(window: Handle, process: *mut u32) -> u32;
    fn SetWindowPos(window: Handle, after: Handle, x: i32, y: i32, w: i32, h: i32, flags: u32) -> Bool;
    fn SetForegroundWindow(window: Handle) -> Bool;
    fn GetDC(window: Handle) -> Handle;
    fn ReleaseDC(window: Handle, dc: Handle) -> i32;
    fn SetCursorPos(x: i32, y: i32) -> Bool;
    fn GetCursorPos(point: *mut Point) -> Bool;
    fn SendInput(count: u32, inputs: *const Input, size: i32) -> u32;
}

#[link(name = "gdi32")]
extern "system" {
    fn CreateCompatibleDC(dc: Handle) -> Handle;
    fn CreateCompatibleBitmap(dc: Handle, w: i32, h: i32) -> Handle;
    fn SelectObject(dc: Handle, object: Handle) -> Handle;
    fn BitBlt(dst: Handle, x: i32, y: i32, w: i32, h: i32, src: Handle, sx: i32, sy: i32, rop: u32) -> Bool;
    fn GetDIBits(dc: Handle, bitmap: Handle, start: u32, lines: u32, bits: *mut u8, info: *mut BitmapInfo, usage: u32) -> i32;
    fn DeleteObject(object: Handle) -> Bool;
    fn DeleteDC(dc: Handle) -> Bool;
}

/// A window the bot can read and drive.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Window {
    handle: Handle,
}

// The handle is an opaque OS identifier, not a pointer this crate dereferences.
unsafe impl Send for Window {}

impl fmt::Debug for Window {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Window({:?}, {:?})", self.title(), self.size())
    }
}

/// Collected during enumeration, since the callback cannot capture state.
static mut FOUND: Vec<(Handle, u32)> = Vec::new();

extern "system" fn collect(window: Handle, _param: isize) -> Bool {
    // SAFETY: EnumWindows runs the callback synchronously on this thread, so
    // the static is not shared across threads for the duration of the call.
    unsafe {
        if IsWindowVisible(window) != 0 {
            let mut process = 0u32;
            GetWindowThreadProcessId(window, &mut process);
            #[allow(static_mut_refs)]
            FOUND.push((window, process));
        }
    }
    1
}

impl Window {
    /// Every visible top-level window belonging to a process whose executable
    /// name contains `needle`, largest first.
    ///
    /// Selecting by process rather than title matters here: the client renames
    /// its window to the club and stakes of whichever table is open, so a title
    /// match would need updating for every table. The executable does not
    /// change.
    pub fn find_by_process(needle: &str) -> Vec<Window> {
        let wanted = process_ids_named(needle);
        // SAFETY: single-threaded enumeration into a static, cleared first.
        let windows: Vec<(Handle, u32)> = unsafe {
            #[allow(static_mut_refs)]
            {
                FOUND.clear();
                EnumWindows(collect, 0);
                FOUND.clone()
            }
        };

        let mut matches: Vec<Window> = windows
            .into_iter()
            .filter(|(_, process)| wanted.contains(process))
            .map(|(handle, _)| Window { handle })
            .filter(|w| {
                let (width, height) = w.size();
                width > 0 && height > 0
            })
            .collect();
        matches.sort_by_key(|w| {
            let (width, height) = w.size();
            std::cmp::Reverse(width as i64 * height as i64)
        });
        matches
    }

    /// The window's title, which the client changes per table.
    pub fn title(&self) -> String {
        let mut buffer = [0u16; 512];
        // SAFETY: the buffer outlives the call and its length is passed.
        let len = unsafe { GetWindowTextW(self.handle, buffer.as_mut_ptr(), buffer.len() as i32) };
        String::from_utf16_lossy(&buffer[..len.max(0) as usize])
    }

    /// Outer size in pixels, as `(width, height)`.
    pub fn size(&self) -> (usize, usize) {
        let rect = self.rect();
        (
            (rect.right - rect.left).max(0) as usize,
            (rect.bottom - rect.top).max(0) as usize,
        )
    }

    /// Top-left corner on the virtual desktop.
    pub fn position(&self) -> (i32, i32) {
        let rect = self.rect();
        (rect.left, rect.top)
    }

    fn rect(&self) -> Rect {
        let mut rect = Rect::default();
        // SAFETY: `rect` is a valid, correctly sized output pointer.
        unsafe { GetWindowRect(self.handle, &mut rect) };
        rect
    }

    /// Resizes the window without moving it.
    ///
    /// Templates are pixel-exact, so this must succeed before any reading. It
    /// returns the size actually achieved, which can differ if the client
    /// enforces a minimum or an aspect ratio — the caller must check rather
    /// than assume.
    pub fn resize(&self, width: usize, height: usize) -> (usize, usize) {
        // SAFETY: a null "insert after" handle with SWP_NOZORDER leaves the
        // z-order alone, and SWP_NOMOVE ignores the position arguments.
        unsafe {
            SetWindowPos(
                self.handle,
                std::ptr::null_mut(),
                0,
                0,
                width as i32,
                height as i32,
                SWP_NOMOVE | SWP_NOZORDER,
            );
        }
        self.size()
    }

    /// Brings the window to the front. Capture reads the screen, so anything
    /// covering the window would be captured instead of it.
    pub fn focus(&self) -> bool {
        // SAFETY: the handle came from enumeration and is checked by the OS.
        unsafe { SetForegroundWindow(self.handle) != 0 }
    }

    /// Captures the window as raw RGB, row-major, three bytes per pixel.
    ///
    /// Returns `None` if the window has no area or the OS refuses the copy.
    pub fn capture(&self) -> Option<Capture> {
        let (width, height) = self.size();
        if width == 0 || height == 0 {
            return None;
        }
        let (w, h) = (width as i32, height as i32);

        // SAFETY: every handle is checked before use and released on all paths
        // below, including the early return.
        unsafe {
            let screen = GetDC(std::ptr::null_mut());
            if screen.is_null() {
                return None;
            }
            let memory = CreateCompatibleDC(screen);
            let bitmap = CreateCompatibleBitmap(screen, w, h);
            let mut pixels = vec![0u8; width * height * 4];
            let mut ok = false;

            if !memory.is_null() && !bitmap.is_null() {
                let previous = SelectObject(memory, bitmap);
                let (x, y) = self.position();
                if BitBlt(memory, 0, 0, w, h, screen, x, y, SRCCOPY) != 0 {
                    let mut info = BitmapInfo {
                        header: BitmapInfoHeader {
                            size: std::mem::size_of::<BitmapInfoHeader>() as u32,
                            width: w,
                            // Negative height requests a top-down image, so row
                            // 0 is the top rather than the bottom.
                            height: -h,
                            planes: 1,
                            bit_count: 32,
                            compression: BI_RGB,
                            ..Default::default()
                        },
                        colors: [0; 3],
                    };
                    ok = GetDIBits(
                        memory,
                        bitmap,
                        0,
                        height as u32,
                        pixels.as_mut_ptr(),
                        &mut info,
                        DIB_RGB_COLORS,
                    ) != 0;
                }
                SelectObject(memory, previous);
            }

            if !bitmap.is_null() {
                DeleteObject(bitmap);
            }
            if !memory.is_null() {
                DeleteDC(memory);
            }
            ReleaseDC(std::ptr::null_mut(), screen);

            if !ok {
                return None;
            }
            // Windows hands back BGRA; the recogniser wants RGB.
            let mut rgb = Vec::with_capacity(width * height * 3);
            for chunk in pixels.chunks_exact(4) {
                rgb.push(chunk[2]);
                rgb.push(chunk[1]);
                rgb.push(chunk[0]);
            }
            Some(Capture {
                width,
                height,
                rgb,
            })
        }
    }

    /// Replaces the contents of a focused text box with `text`.
    ///
    /// Used for the bet amount, where the client's preset buttons offer only a
    /// few sizes and a solved strategy means a specific one. A blueprint that
    /// decided to raise meant an amount, and clicking the nearest preset
    /// instead plays a strategy nobody solved for.
    ///
    /// Characters are sent as Unicode rather than as key codes, so whatever
    /// keyboard layout is installed cannot change what arrives — a scan code
    /// for `2` is a different character on a French layout, a Unicode `2`
    /// never is.
    ///
    /// Returns false if the system declined any event. As with clicking, that
    /// is not the same as the client having accepted it: the caller must read
    /// the box back and see what actually landed.
    pub fn type_text(&self, text: &str) -> bool {
        let events = text_events(text);
        // SAFETY: `events` is a correctly sized array of correctly laid out
        // structures, and the count matches its length.
        unsafe {
            SendInput(
                events.len() as u32,
                events.as_ptr(),
                std::mem::size_of::<Input>() as i32,
            ) == events.len() as u32
        }
    }

    /// Clicks a point given relative to the window's top-left corner.
    ///
    /// Returns false if the cursor could not be placed, which is a different
    /// failure from the application ignoring the click and needs to be
    /// distinguished by the caller.
    pub fn click_at(&self, x: usize, y: usize) -> bool {
        let (left, top) = self.position();
        let (target_x, target_y) = (left + x as i32, top + y as i32);

        // SAFETY: all arguments are plain integers or a correctly sized array.
        unsafe {
            if SetCursorPos(target_x, target_y) == 0 {
                return false;
            }
            let mut landed = Point::default();
            GetCursorPos(&mut landed);
            if landed.x != target_x || landed.y != target_y {
                return false;
            }

            let events = [
                mouse_event(MOUSEEVENTF_LEFTDOWN),
                mouse_event(MOUSEEVENTF_LEFTUP),
            ];
            SendInput(
                events.len() as u32,
                events.as_ptr(),
                std::mem::size_of::<Input>() as i32,
            ) == events.len() as u32
        }
    }
}

/// The events that replace a text box's contents with `text`.
///
/// Built separately from sending them so the sequence can be checked without a
/// window: a short or malformed sequence would put half an amount in the box,
/// and `1` where `18.7` was meant is a bet rather than a typo.
fn text_events(text: &str) -> Vec<Input> {
    let mut events = Vec::with_capacity(text.len() * 2 + 6);
    // Select whatever is there and delete it, so this replaces rather than
    // appends. Appending to a box already reading "2" would bet 23.
    events.extend(chord(VK_CONTROL, VK_A));
    events.extend(stroke(VK_BACK));
    for unit in text.encode_utf16() {
        for release in [false, true] {
            events.push(Input {
                kind: INPUT_KEYBOARD,
                value: InputValue {
                    keyboard: KeyboardInput {
                        virtual_key: 0,
                        scan_code: unit,
                        flags: KEYEVENTF_UNICODE | if release { KEYEVENTF_KEYUP } else { 0 },
                        time: 0,
                        extra: 0,
                    },
                },
            });
        }
    }
    events
}

/// One mouse event at the cursor's current position.
fn mouse_event(flags: u32) -> Input {
    Input {
        kind: INPUT_MOUSE,
        value: InputValue {
            mouse: MouseInput {
                dx: 0,
                dy: 0,
                mouse_data: 0,
                flags,
                time: 0,
                extra: 0,
            },
        },
    }
}

/// Press and release one key.
fn stroke(key: u16) -> Vec<Input> {
    [false, true]
        .into_iter()
        .map(|release| key_event(key, release))
        .collect()
}

/// Hold one key, press another, release both.
fn chord(modifier: u16, key: u16) -> Vec<Input> {
    vec![
        key_event(modifier, false),
        key_event(key, false),
        key_event(key, true),
        key_event(modifier, true),
    ]
}

fn key_event(key: u16, release: bool) -> Input {
    Input {
        kind: INPUT_KEYBOARD,
        value: InputValue {
            keyboard: KeyboardInput {
                virtual_key: key,
                scan_code: 0,
                flags: if release { KEYEVENTF_KEYUP } else { 0 },
                time: 0,
                extra: 0,
            },
        },
    }
}

/// A captured frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Capture {
    pub width: usize,
    pub height: usize,
    /// Row-major RGB, three bytes per pixel.
    pub rgb: Vec<u8>,
}

impl Capture {
    /// Whether the frame is a single flat colour.
    ///
    /// An application that blocks screen capture returns exactly this, and it
    /// must never be mistaken for a real frame — a recogniser handed a blank
    /// image reports "no cards found", which reads as an empty table.
    pub fn is_blank(&self) -> bool {
        if self.rgb.len() < 3 {
            return true;
        }
        let first = &self.rgb[0..3];
        self.rgb.chunks_exact(3).all(|p| p == first)
    }
}

/// Process ids whose executable name contains `needle`, case-insensitively.
fn process_ids_named(needle: &str) -> Vec<u32> {
    // Reading the process list without another dependency: the OS exposes it
    // through the toolhelp snapshot API.
    #[repr(C)]
    struct ProcessEntry {
        size: u32,
        usage: u32,
        id: u32,
        default_heap: usize,
        module_id: u32,
        threads: u32,
        parent: u32,
        priority: i32,
        flags: u32,
        exe_file: [u16; 260],
    }
    const SNAP_PROCESS: u32 = 0x0000_0002;
    const INVALID: isize = -1;

    #[link(name = "kernel32")]
    extern "system" {
        fn CreateToolhelp32Snapshot(flags: u32, process: u32) -> Handle;
        fn Process32FirstW(snapshot: Handle, entry: *mut ProcessEntry) -> Bool;
        fn Process32NextW(snapshot: Handle, entry: *mut ProcessEntry) -> Bool;
        fn CloseHandle(object: Handle) -> Bool;
    }

    let needle = needle.to_lowercase();
    let mut ids = Vec::new();
    // SAFETY: the snapshot is closed on every path, and the entry struct is
    // sized before each call as the API requires.
    unsafe {
        let snapshot = CreateToolhelp32Snapshot(SNAP_PROCESS, 0);
        if snapshot as isize == INVALID || snapshot.is_null() {
            return ids;
        }
        let mut entry: ProcessEntry = std::mem::zeroed();
        entry.size = std::mem::size_of::<ProcessEntry>() as u32;
        let mut more = Process32FirstW(snapshot, &mut entry);
        while more != 0 {
            let end = entry
                .exe_file
                .iter()
                .position(|c| *c == 0)
                .unwrap_or(entry.exe_file.len());
            let name = String::from_utf16_lossy(&entry.exe_file[..end]).to_lowercase();
            if name.contains(&needle) {
                ids.push(entry.id);
            }
            entry.size = std::mem::size_of::<ProcessEntry>() as u32;
            more = Process32NextW(snapshot, &mut entry);
        }
        CloseHandle(snapshot);
    }
    ids
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_blank_capture_is_recognised_as_blank() {
        // The shape a capture-blocking application returns. Mistaking it for a
        // real frame would read as an empty table rather than a failure.
        let blank = Capture {
            width: 2,
            height: 2,
            rgb: vec![7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
        };
        assert!(blank.is_blank());

        let real = Capture {
            width: 2,
            height: 2,
            rgb: vec![7, 7, 7, 7, 7, 7, 7, 7, 7, 9, 9, 9],
        };
        assert!(!real.is_blank());
    }

    #[test]
    fn an_empty_capture_counts_as_blank() {
        let empty = Capture {
            width: 0,
            height: 0,
            rgb: Vec::new(),
        };
        assert!(empty.is_blank());
    }

    /// The keyboard member has to fit the union the API expects.
    ///
    /// `SendInput` is told one size for every event, so a keyboard event has to
    /// occupy the same `INPUT` as a mouse event. Getting this wrong does not
    /// crash — the call reports fewer events accepted, or the wrong keys
    /// arrive, both of which look like the application ignoring input.
    #[test]
    fn a_keyboard_event_fits_the_same_input_as_a_mouse_event() {
        assert_eq!(std::mem::size_of::<KeyboardInput>(), 24);
        assert_eq!(std::mem::size_of::<InputValue>(), 32, "the union is as wide as its widest member");
        assert_eq!(std::mem::size_of::<Input>(), 40, "what SendInput is told");
        assert_eq!(
            std::mem::align_of::<InputValue>(),
            std::mem::align_of::<MouseInput>()
        );
    }

    #[test]
    fn typing_an_amount_clears_the_box_before_writing_to_it() {
        let events = text_events("18.7");
        // Four to hold control and press A, two to delete, then a press and a
        // release for each of the four characters.
        assert_eq!(events.len(), 4 + 2 + 8);

        // SAFETY: every event here was built as a keyboard event.
        unsafe {
            assert_eq!(events[0].value.keyboard.virtual_key, VK_CONTROL);
            assert_eq!(events[1].value.keyboard.virtual_key, VK_A);
            assert_eq!(events[4].value.keyboard.virtual_key, VK_BACK);
            // The characters go as Unicode, so no virtual key at all.
            assert_eq!(events[6].value.keyboard.virtual_key, 0);
            assert_eq!(events[6].value.keyboard.scan_code, u16::from(b'1'));
            assert_eq!(events[6].value.keyboard.flags & KEYEVENTF_UNICODE, KEYEVENTF_UNICODE);
        }
        assert!(events.iter().all(|e| e.kind == INPUT_KEYBOARD));
    }

    #[test]
    fn an_empty_amount_still_clears_the_box() {
        // Otherwise a failed read could leave the previous bet sitting there.
        assert_eq!(text_events("").len(), 6);
    }

    #[test]
    fn the_input_struct_matches_what_the_api_expects() {
        // A wrong size here makes SendInput silently accept zero events, which
        // would look exactly like the application filtering the click.
        assert_eq!(std::mem::size_of::<Input>(), 40, "INPUT is 40 bytes on x64");
        assert_eq!(std::mem::size_of::<MouseInput>(), 32);
        assert_eq!(std::mem::align_of::<Input>(), 8);
    }

    #[test]
    fn the_bitmap_header_matches_what_the_api_expects() {
        assert_eq!(std::mem::size_of::<BitmapInfoHeader>(), 40);
    }

    #[test]
    fn searching_for_a_process_that_is_not_running_finds_nothing() {
        assert!(process_ids_named("a-process-that-does-not-exist-xyzzy").is_empty());
        assert!(Window::find_by_process("a-process-that-does-not-exist-xyzzy").is_empty());
    }

    #[test]
    fn the_current_process_is_findable_by_name() {
        // Exercises the snapshot walk against something guaranteed present.
        let ids = process_ids_named("poker");
        assert!(
            !ids.is_empty(),
            "the test binary's own name should contain 'poker'"
        );
    }
}
