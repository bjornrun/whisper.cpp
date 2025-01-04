# Whisper GStreamer Sink Plugin

This is a GStreamer sink plugin that performs real-time speech recognition using Whisper. The plugin accepts raw audio input and outputs transcribed text to stdout.

## Building

To build the plugin, you need GStreamer development files installed:

```bash
# On Debian/Ubuntu
sudo apt-get install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev

# On Fedora
sudo dnf install gstreamer1-devel gstreamer1-plugins-base-devel

# On macOS
brew install gstreamer gst-plugins-base
```

Then build with CMake:

```bash
cmake -B build
cmake --build build
```

## Usage

After building, you can use the plugin in GStreamer pipelines. Here are some examples:

1. Transcribe from microphone:
```bash
gst-launch-1.0 pulsesrc ! audioconvert ! audioresample ! audio/x-raw,format=F32LE,channels=1,rate=16000 ! whispersink model-path=/path/to/model.bin
```

2. Transcribe from file:
```bash
gst-launch-1.0 filesrc location=audio.wav ! wavparse ! audioconvert ! audioresample ! audio/x-raw,format=F32LE,channels=1,rate=16000 ! whispersink model-path=/path/to/model.bin
```

## Properties

The plugin supports the following properties:

- `model-path`: Path to the Whisper model file (default: "models/ggml-base.en.bin")
- `n-threads`: Number of threads to use for processing (default: 4)
- `translate`: Whether to translate to English (default: false)
- `language`: Source language code (default: "en")
- `use-gpu`: Whether to use GPU for inference (default: true)

Example with properties:
```bash
gst-launch-1.0 pulsesrc ! audioconvert ! audioresample ! audio/x-raw,format=F32LE,channels=1,rate=16000 ! whispersink model-path=/path/to/model.bin language=fr translate=true n-threads=8
``` 