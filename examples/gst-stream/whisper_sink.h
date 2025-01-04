#ifndef WHISPER_SINK_H
#define WHISPER_SINK_H

#include <gst/gst.h>
#include <gst/audio/audio.h>
#include <gst/base/gstbasesink.h>
#include <vector>
#include "whisper.h"

G_BEGIN_DECLS

#define WHISPER_TYPE_SINK (whisper_sink_get_type())
#define WHISPER_SINK(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj),WHISPER_TYPE_SINK,WhisperSink))
#define WHISPER_SINK_CLASS(klass) (G_TYPE_CHECK_CLASS_CAST((klass),WHISPER_TYPE_SINK,WhisperSinkClass))
#define WHISPER_IS_SINK(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj),WHISPER_TYPE_SINK))
#define WHISPER_IS_SINK_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass),WHISPER_TYPE_SINK))

typedef struct _WhisperSink WhisperSink;
typedef struct _WhisperSinkClass WhisperSinkClass;

struct _WhisperSink {
    GstBaseSink parent;

    // Properties
    gchar *model_path;
    gint n_threads;
    gboolean translate;
    gchar *language;
    gboolean use_gpu;
    gfloat vad_thold;
    gfloat freq_thold;

    // State
    struct whisper_context *ctx;
    GstAudioInfo audio_info;
    GMutex mutex;
    
    // Buffer management
    float *audio_buffer;
    size_t buffer_size;
    size_t samples_collected;

    gint step_ms;
    gint length_ms;
    gint keep_ms;
    std::vector<whisper_token> prompt_tokens;

    gint n_samples_step;
    gint n_samples_len;
    gint n_samples_keep;
    gint n_samples_30s;

    int64_t t1_last;
};

struct _WhisperSinkClass {
    GstBaseSinkClass parent_class;
};

GType whisper_sink_get_type(void);

G_END_DECLS

#endif // WHISPER_SINK_H 