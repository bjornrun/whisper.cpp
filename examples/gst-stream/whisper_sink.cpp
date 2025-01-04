#include "whisper_sink.h"
#include "common.h"
#include <string>
#include <vector>
#include <cstring>  // for memcpy

#define PACKAGE "whisper-gst"
#define PACKAGE_VERSION "1.0"

#define WHISPER_SAMPLE_RATE 16000
#define DEFAULT_MODEL_PATH "models/ggml-base.en.bin"
#define DEFAULT_LANGUAGE "en"
#define DEFAULT_N_THREADS 4
#define DEFAULT_USE_GPU TRUE
#define BUFFER_SIZE (WHISPER_SAMPLE_RATE * 30) // 30 seconds buffer

GST_DEBUG_CATEGORY_STATIC(whisper_sink_debug);
#define GST_CAT_DEFAULT whisper_sink_debug

enum {
    PROP_0,
    PROP_MODEL_PATH,
    PROP_N_THREADS,
    PROP_TRANSLATE,
    PROP_LANGUAGE,
    PROP_USE_GPU,
    PROP_STEP_MS,
    PROP_LENGTH_MS,
    PROP_KEEP_MS,
    PROP_VAD_THOLD,
    PROP_FREQ_THOLD
};

static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE(
    "sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS("audio/x-raw, "
                    "format = (string) F32LE, "
                    "rate = (int) " G_STRINGIFY(WHISPER_SAMPLE_RATE) ", "
                    "channels = (int) 1")
);

#define whisper_sink_parent_class parent_class
G_DEFINE_TYPE(WhisperSink, whisper_sink, GST_TYPE_BASE_SINK);

static void whisper_sink_set_property(GObject *object, guint prop_id,
                                    const GValue *value, GParamSpec *pspec);
static void whisper_sink_get_property(GObject *object, guint prop_id,
                                    GValue *value, GParamSpec *pspec);
static void whisper_sink_finalize(GObject *object);
static gboolean whisper_sink_start(GstBaseSink *sink);
static gboolean whisper_sink_stop(GstBaseSink *sink);
static GstFlowReturn whisper_sink_render(GstBaseSink *sink, GstBuffer *buffer);

static void whisper_sink_class_init(WhisperSinkClass *klass) {
    GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
    GstBaseSinkClass *base_sink_class = GST_BASE_SINK_CLASS(klass);
    GstElementClass *element_class = GST_ELEMENT_CLASS(klass);

    gobject_class->set_property = whisper_sink_set_property;
    gobject_class->get_property = whisper_sink_get_property;
    gobject_class->finalize = whisper_sink_finalize;

    base_sink_class->start = whisper_sink_start;
    base_sink_class->stop = whisper_sink_stop;
    base_sink_class->render = whisper_sink_render;

    g_object_class_install_property(gobject_class, PROP_MODEL_PATH,
        g_param_spec_string("model-path", "Model Path",
            "Path to the Whisper model file", DEFAULT_MODEL_PATH,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(gobject_class, PROP_N_THREADS,
        g_param_spec_int("n-threads", "Number of Threads",
            "Number of threads to use for processing", 1, G_MAXINT, DEFAULT_N_THREADS,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(gobject_class, PROP_TRANSLATE,
        g_param_spec_boolean("translate", "Translate",
            "Translate to English", FALSE,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(gobject_class, PROP_LANGUAGE,
        g_param_spec_string("language", "Language",
            "Source language code", DEFAULT_LANGUAGE,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(gobject_class, PROP_USE_GPU,
        g_param_spec_boolean("use-gpu", "Use GPU",
            "Use GPU for inference", DEFAULT_USE_GPU,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(gobject_class, PROP_STEP_MS,
        g_param_spec_int("step-ms", "Step Milliseconds",
            "Audio step size in milliseconds", 1, G_MAXINT, 3000,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(gobject_class, PROP_LENGTH_MS,
        g_param_spec_int("length-ms", "Length Milliseconds",
            "Audio length in milliseconds", 1, G_MAXINT, 10000,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(gobject_class, PROP_KEEP_MS,
        g_param_spec_int("keep-ms", "Keep Milliseconds",
            "Audio to keep from previous step in milliseconds", 0, G_MAXINT, 200,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(gobject_class, PROP_VAD_THOLD,
        g_param_spec_float("vad-thold", "VAD Threshold",
            "Voice Activity Detection threshold", 0.0, 1.0, 0.6f,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(gobject_class, PROP_FREQ_THOLD,
        g_param_spec_float("freq-thold", "Frequency Threshold",
            "Frequency threshold for high-pass filtering", 0.0, 1000.0, 100.0f,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    gst_element_class_set_static_metadata(element_class,
        "Whisper Speech-to-Text Sink", "Sink/Audio",
        "Performs real-time speech-to-text using Whisper",
        "Bjorn Runaker <bjornrun@gmail.com>");

    gst_element_class_add_static_pad_template(element_class, &sink_template);

    GST_DEBUG_CATEGORY_INIT(whisper_sink_debug, "whispersink", 0,
        "Whisper Speech-to-Text Sink");
}

static void whisper_sink_init(WhisperSink *sink) {
    sink->model_path = g_strdup(DEFAULT_MODEL_PATH);
    sink->n_threads = DEFAULT_N_THREADS;
    sink->translate = FALSE;
    sink->language = g_strdup(DEFAULT_LANGUAGE);
    sink->use_gpu = DEFAULT_USE_GPU;
    sink->ctx = nullptr;
    sink->audio_buffer = nullptr;
    sink->buffer_size = BUFFER_SIZE;
    sink->samples_collected = 0;
    sink->prompt_tokens.clear();
    sink->step_ms = 3000;
    sink->length_ms = 10000;
    sink->keep_ms = 200;

    sink->keep_ms = std::min(sink->keep_ms, sink->step_ms);
    sink->length_ms = std::max(sink->length_ms, sink->step_ms);

    sink->n_samples_step = (1e-3 * sink->step_ms) * WHISPER_SAMPLE_RATE;
    sink->n_samples_len = (1e-3 * sink->length_ms) * WHISPER_SAMPLE_RATE;
    sink->n_samples_keep = (1e-3 * sink->keep_ms) * WHISPER_SAMPLE_RATE;
    sink->n_samples_30s = (1e-3 * 30000.0) * WHISPER_SAMPLE_RATE;

    sink->t1_last = 0;

    g_mutex_init(&sink->mutex);

    sink->vad_thold = 0.6f;
    sink->freq_thold = 100.0f;
}

static int vad_end(const float *buffer, int length, int sample_rate, int last_ms, float vad_thold, float freq_thold, bool verbose)
{
    std::vector<float> pcmf32(buffer, buffer + length);
    const int n_samples = length;
    const int n_samples_last = (sample_rate * last_ms) / 1000;

    if (n_samples_last >= n_samples)
    {
        // not enough samples
        return -1;
    }

    if (freq_thold > 0.0f)
    {      
        high_pass_filter(pcmf32, freq_thold, sample_rate);
    }

    float energy_all = 0.0f;
    for (int i = 0; i < n_samples; i++)
    {
        energy_all += fabsf(pcmf32[i]);
    }
    energy_all /= n_samples;

    // Search for the last window of length n_samples_last with low energy
    for (int start = n_samples - n_samples_last; start >= 0; start--)
    {
        float energy_window = 0.0f;
        for (int i = start; i < start + n_samples_last; i++)
        {
            energy_window += fabsf(pcmf32[i]);
        }
        energy_window /= n_samples_last;

        if (verbose)
        {
            fprintf(stderr, "%s: energy_all: %f, energy_window: %f, vad_thold: %f, freq_thold: %f\n", 
                    __func__, energy_all, energy_window, vad_thold, freq_thold);
        }

        if (energy_window <= vad_thold * energy_all)
        {
            return start;
        }
    }

    return -1;
}

static void whisper_sink_process_audio(WhisperSink *sink) {
    if (sink->samples_collected < WHISPER_SAMPLE_RATE) {
        return; // Not enough samples to process
    }

    g_mutex_lock(&sink->mutex);

    whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wparams.print_progress = false;
    wparams.print_special = false;
    wparams.print_realtime = false;
    wparams.print_timestamps = true;
    wparams.translate = sink->translate;
    wparams.language = sink->language;
    wparams.n_threads = sink->n_threads;
    
    wparams.prompt_tokens = sink->prompt_tokens.empty() ? nullptr : sink->prompt_tokens.data();
    wparams.prompt_n_tokens = sink->prompt_tokens.size();

    int vad_end_idx = vad_end(sink->audio_buffer, sink->samples_collected,
                              WHISPER_SAMPLE_RATE, sink->keep_ms, sink->vad_thold, sink->freq_thold, false);
    int n_samples_to_process = vad_end_idx >= 0 ? vad_end_idx : sink->samples_collected;
    if (n_samples_to_process > sink->n_samples_step) {
        if (whisper_full(sink->ctx, wparams, sink->audio_buffer, n_samples_to_process) != 0)
        {
            GST_ERROR_OBJECT(sink, "Failed to process audio");
            g_mutex_unlock(&sink->mutex);
            return;
        }

        // Print results
        const int n_segments = whisper_full_n_segments(sink->ctx);
        for (int i = 0; i < n_segments; ++i)
        {
            const char *text = whisper_full_get_segment_text(sink->ctx, i);
            const int64_t t0 = whisper_full_get_segment_t0(sink->ctx, i) + sink->t1_last;
            const int64_t t1 = whisper_full_get_segment_t1(sink->ctx, i) + sink->t1_last;
            sink->t1_last = t1;
            std::string output = "[" + to_timestamp(t0, false) + " --> " + to_timestamp(t1, false) + "]  " + text;
            g_print("%s\n", output.c_str());
            // printf("[%d:%02d.%03d --> %d:%02d.%03d] %s\n",
            //     (int)(t0 / 60000), (int)((t0 / 1000) % 60), (int)(t0 % 1000),
            //     (int)(t1 / 60000), (int)((t1 / 1000) % 60), (int)(t1 % 1000),
            //     text);
        }

        // Save last n_samples_keep samples
        if (sink->samples_collected - n_samples_to_process > 0)
        {
            // int n_samples_to_keep = sink->n_samples_keep < sink->samples_collected ? sink->n_samples_keep : sink->samples_collected;
            int n_samples_unprocessed = sink->samples_collected - n_samples_to_process;
            memmove(sink->audio_buffer,
                    sink->audio_buffer + n_samples_to_process,
                    n_samples_unprocessed * sizeof(float));
            sink->samples_collected = n_samples_unprocessed;
        }
        else
        {
            sink->samples_collected = 0;
        }

        sink->prompt_tokens.clear();

        for (int i = 0; i < n_segments; ++i)
        {
            const int token_count = whisper_full_n_tokens(sink->ctx, i);
            for (int j = 0; j < token_count; ++j)
            {
                sink->prompt_tokens.push_back(whisper_full_get_token_id(sink->ctx, i, j));
            }
        }
    }
    g_mutex_unlock(&sink->mutex);
}

static GstFlowReturn whisper_sink_render(GstBaseSink *base_sink, GstBuffer *buffer) {
    WhisperSink *sink = WHISPER_SINK(base_sink);
    
    GstMapInfo map;
    if (!gst_buffer_map(buffer, &map, GST_MAP_READ)) {
        GST_ERROR_OBJECT(sink, "Failed to map buffer");
        return GST_FLOW_ERROR;
    }

    const float *samples = (const float *)map.data;
    const size_t n_samples = map.size / sizeof(float);

    g_mutex_lock(&sink->mutex);
    
    // Copy new samples to buffer
    if (sink->samples_collected + n_samples <= sink->buffer_size) {
        memcpy(sink->audio_buffer + sink->samples_collected, samples, n_samples * sizeof(float));
        sink->samples_collected += n_samples;
    }
    
    g_mutex_unlock(&sink->mutex);
    
    gst_buffer_unmap(buffer, &map);

    if (sink->samples_collected >= sink->n_samples_step) {
        // Process audio if we have collected enough samples
        whisper_sink_process_audio(sink);
    }

    return GST_FLOW_OK;
}

static gboolean whisper_sink_start(GstBaseSink *base_sink) {
    WhisperSink *sink = WHISPER_SINK(base_sink);

    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = sink->use_gpu;
    
    sink->ctx = whisper_init_from_file_with_params(sink->model_path, cparams);
    if (!sink->ctx) {
        GST_ERROR_OBJECT(sink, "Failed to initialize Whisper context");
        return FALSE;
    }

    sink->audio_buffer = (float *)g_malloc0(sink->buffer_size * sizeof(float));
    if (!sink->audio_buffer) {
        GST_ERROR_OBJECT(sink, "Failed to allocate audio buffer");
        whisper_free(sink->ctx);
        sink->ctx = nullptr;
        return FALSE;
    }

    return TRUE;
}

static gboolean whisper_sink_stop(GstBaseSink *base_sink) {
    WhisperSink *sink = WHISPER_SINK(base_sink);

    if (sink->ctx) {
        whisper_free(sink->ctx);
        sink->ctx = nullptr;
    }

    g_free(sink->audio_buffer);
    sink->audio_buffer = nullptr;

    return TRUE;
}

static void whisper_sink_finalize(GObject *object) {
    WhisperSink *sink = WHISPER_SINK(object);

    g_free(sink->model_path);
    g_free(sink->language);
    g_mutex_clear(&sink->mutex);

    G_OBJECT_CLASS(parent_class)->finalize(object);
}

static void whisper_sink_set_property(GObject *object, guint prop_id,
                                    const GValue *value, GParamSpec *pspec) {
    WhisperSink *sink = WHISPER_SINK(object);

    switch (prop_id) {
        case PROP_MODEL_PATH:
            g_free(sink->model_path);
            sink->model_path = g_value_dup_string(value);
            break;
        case PROP_N_THREADS:
            sink->n_threads = g_value_get_int(value);
            break;
        case PROP_TRANSLATE:
            sink->translate = g_value_get_boolean(value);
            break;
        case PROP_LANGUAGE:
            g_free(sink->language);
            sink->language = g_value_dup_string(value);
            break;
        case PROP_USE_GPU:
            sink->use_gpu = g_value_get_boolean(value);
            break;
        case PROP_STEP_MS:
            sink->step_ms = g_value_get_int(value);
            break;
        case PROP_LENGTH_MS:
            sink->length_ms = g_value_get_int(value);
            break;
        case PROP_KEEP_MS:
            sink->keep_ms = g_value_get_int(value);
            break;
        case PROP_VAD_THOLD:
            sink->vad_thold = g_value_get_float(value);
            break;
        case PROP_FREQ_THOLD:
            sink->freq_thold = g_value_get_float(value);
            break;
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
            break;
    }
}

static void whisper_sink_get_property(GObject *object, guint prop_id,
                                    GValue *value, GParamSpec *pspec) {
    WhisperSink *sink = WHISPER_SINK(object);

    switch (prop_id) {
        case PROP_MODEL_PATH:
            g_value_set_string(value, sink->model_path);
            break;
        case PROP_N_THREADS:
            g_value_set_int(value, sink->n_threads);
            break;
        case PROP_TRANSLATE:
            g_value_set_boolean(value, sink->translate);
            break;
        case PROP_LANGUAGE:
            g_value_set_string(value, sink->language);
            break;
        case PROP_USE_GPU:
            g_value_set_boolean(value, sink->use_gpu);
            break;
        case PROP_STEP_MS:
            g_value_set_int(value, sink->step_ms);
            break;
        case PROP_LENGTH_MS:
            g_value_set_int(value, sink->length_ms);
            break;
        case PROP_KEEP_MS:
            g_value_set_int(value, sink->keep_ms);
            break;
        case PROP_VAD_THOLD:
            g_value_set_float(value, sink->vad_thold);
            break;
        case PROP_FREQ_THOLD:
            g_value_set_float(value, sink->freq_thold);
            break;
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
            break;
    }
}

static gboolean plugin_init(GstPlugin *plugin) {
    GST_DEBUG_CATEGORY_INIT(whisper_sink_debug, "whispersink", 0,
        "Whisper Speech-to-Text Sink");
    
    gboolean ret = gst_element_register(plugin, "whispersink", GST_RANK_NONE,
                              WHISPER_TYPE_SINK);
    
    g_print("Whisper sink plugin initialization %s\n", ret ? "succeeded" : "failed");
    return ret;
}

GST_PLUGIN_DEFINE(
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    whispersink,
    "Whisper Speech-to-Text Sink",
    plugin_init,
    PACKAGE_VERSION,
    "LGPL",
    PACKAGE,
    "https://github.com/ggerganov/whisper.cpp"
) 