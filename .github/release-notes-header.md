# Quick install — first time setup

One laptop runs the instructor console; one Android phone watches each
trainee. Set the laptop up first so the phones have something to find.

## 1. Instructor console (the laptop)

Download the binary for your OS from the assets below, then:

**macOS** (Apple Silicon)
```bash
chmod +x argus-macos-arm64 && ./argus-macos-arm64
```
If macOS blocks it as from an unidentified developer: right-click → Open once,
or `xattr -d com.apple.quarantine argus-macos-arm64`.

**Windows** — run `argus-windows-x64.exe`. If SmartScreen objects: More info →
Run anyway. **Linux** — `chmod +x argus-linux-x64 && ./argus-linux-x64`.

Launched with no arguments it starts the server and opens the console at
`http://localhost:8080/`. Allow it through the firewall when asked — the
phones connect to this machine on port 8765. Run `argus-… doctor` any time to
see the address phones should use and what, if anything, is misconfigured.

## 2. Station app (each phone)

Requires an arm64 Android 12+ phone; the NPU path is built for Snapdragon.

1. Download `argus-edge*.apk` onto the phone and open it — confirm the
   one-time "install unknown apps" prompt (and Play Protect, if it asks).
2. Open **Argus Edge**, pick the floor's use case (e.g. **Fitness**), and
   allow the camera.
3. **Stage the perception models — they are deliberately not inside the APK**
   (licensing: see THIRD-PARTY-NOTICES.md). From a checkout of this repo,
   `python scripts/fetch_edge_models.py` reproduces them byte-identically;
   copy the resulting files to the phone (e.g. into Downloads), then
   **long-press Debug** in the station screen and multi-select them in the
   picker. The status strip reads *Ready* when the model is loaded.

## 3. Connect them

Same Wi-Fi network, then in the phone's **Server** dialog press **Find server
on this network** — the laptop announces itself and the address fills in. On
networks that filter broadcast (common on guest/venue Wi-Fi) type the address
the laptop's `doctor` printed, e.g. `ws://192.168.1.20:8765`. Enter the
trainee's name, pick the exercise (it selects how the laptop scores them),
Connect, then **Start**.

The station appears on the instructor console immediately — grey **ready**
while nobody is in frame, live skeleton and scores once a trainee steps in.
A demo floor with no phones at all: `argus-… demo --ticks 900`, then
`argus-… replay --speed 1.0` against a running server.

> Scope reminder: this is a hackathon build. The triage weights are unfitted
> to real incidents and no accuracy claim is supported — docs/VALIDATION.md
> is the honest list. The APK asset is unsigned unless the repo's signing
> secrets were configured at build time.

---
