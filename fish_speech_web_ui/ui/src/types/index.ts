export interface ProxyConfig {
    commit: {
        first: CommitStageConfig;
        next: CommitStageConfig;
        flush_on_sentence_punctuation: boolean;
        flush_on_clause_punctuation: boolean;
        flush_on_newline: boolean;
        carry_incomplete_tail: boolean;
    };
    tts: {
        reference_id: string;
        max_new_tokens: number;
        chunk_length: number;
        top_p: number;
        repetition_penalty: number;
        temperature: number;
        initial_stream_chunk_size: number;
        stream_chunk_size: number;
    };
    playback: {
        target_emit_bytes: number;
        start_buffer_ms: number;
        stop_grace_ms: number;
    };
    session: {
        max_buffer_chars: number;
        auto_close_on_finish: boolean;
    };
}

export interface CommitStageConfig {
    min_chars: number;
    target_chars: number;
    max_chars: number;
    max_wait_ms: number;
    allow_partial_after_ms: number;
}

export interface SessionOpenResponse {
    ok: boolean;
    session_id: string;
    config: ProxyConfig;
    ttl_sec: number;
}

export interface SessionAppendResponse {
    ok: boolean;
    session_id: string;
    accepted_chars: number;
    buffer_text: string;
    buffer_chars: number;
    committed: CommittedItem[];
    input_closed: boolean;
}

export interface CommittedItem {
    seq: number;
    text: string;
    reason: string;
    created_at: number;
}

export interface StreamEvent {
    type: 'session_start' | 'meta' | 'commit_start' | 'pcm' | 'commit_done' | 'session_done' | 'session_aborted' | 'error';
    session_id: string;
    req_id?: string;
    [key: string]: any;
}
