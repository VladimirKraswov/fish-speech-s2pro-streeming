export interface CommitStageConfig {
  min_chars: number;
  target_chars: number;
  max_chars: number;
  max_wait_ms: number;
  allow_partial_after_ms: number;
}

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
    format?: string;
    normalize?: boolean;
    use_memory_cache?: 'on' | 'off';
    seed?: number | null;
    max_new_tokens: number;
    chunk_length: number;
    top_p: number;
    repetition_penalty: number;
    temperature: number;
    stream_tokens?: boolean;
    initial_stream_chunk_size: number;
    stream_chunk_size: number;
    first_initial_stream_chunk_size?: number | null;
    first_stream_chunk_size?: number | null;
    stateful_synthesis?: boolean;
    stateful_fallback_to_stateless?: boolean;
    stateful_history_turns?: number;
    stateful_history_chars?: number;
    stateful_history_code_frames?: number;
    stateful_reset_every_commits?: number;
    stateful_reset_every_chars?: number;
  };
  playback: {
    target_emit_bytes: number;
    start_buffer_ms: number;
    first_commit_target_emit_bytes?: number;
    first_commit_start_buffer_ms?: number;
    client_start_buffer_ms?: number;
    client_initial_start_delay_ms?: number;
    stop_grace_ms: number;
    boundary_smoothing_enabled?: boolean;
    punctuation_pauses_enabled?: boolean;
    fade_in_ms?: number;
    fade_out_ms?: number;
    pause_after_clause_ms?: number;
    pause_after_sentence_ms?: number;
    pause_after_newline_ms?: number;
    pause_after_force_ms?: number;
    pause_after_hard_limit_ms?: number;
  };
  session: {
    max_buffer_chars: number;
    auto_close_on_finish: boolean;
  };
}

export interface SessionOpenResponse {
  ok: boolean;
  session_id: string;
  config: ProxyConfig;
  ttl_sec: number;
}

export interface CommittedItem {
  seq: number;
  text: string;
  reason: string;
  created_at: number;
}

export interface SessionAppendResponse {
  ok: boolean;
  session_id: string;
  accepted_chars?: number;
  buffer_text?: string;
  buffer_chars?: number;
  committed?: CommittedItem[];
  input_closed?: boolean;
  already_finished?: boolean;
}

export type StreamEvent =
  | {
      type: 'session_start';
      session_id: string;
      req_id?: string;
      target_emit_bytes?: number;
      first_commit_target_emit_bytes?: number;
      first_commit_start_buffer_ms?: number;
      client_start_buffer_ms?: number;
      client_initial_start_delay_ms?: number;
    }
  | {
      type: 'meta';
      session_id?: string;
      commit_seq?: number;
      sample_rate: number;
      channels: number;
      sample_width: number;
    }
  | {
      type: 'commit_start';
      session_id?: string;
      commit_seq: number;
      reason: string;
      text?: string;
      text_preview?: string;
      text_len?: number;
      effective_target_emit_bytes?: number;
      effective_start_buffer_ms?: number;
      server_perf_ms?: number;
    }
  | {
      type: 'pcm';
      session_id?: string;
      commit_seq?: number;
      seq: number;
      data: string;
      first_pcm_for_commit?: boolean;
    }
  | {
      type: 'pause';
      session_id?: string;
      req_id?: string;
      commit_seq?: number;
      boundary?: string;
      pause_ms?: number;
    }
  | {
      type: 'commit_done';
      session_id?: string;
      commit_seq: number;
      upstream_bytes?: number;
      boundary?: string;
      pause_ms?: number;
      fade_in_ms?: number;
      fade_out_ms?: number;
    }
  | {
      type: 'session_done';
      session_id: string;
      req_id?: string;
      commit_count?: number;
    }
  | {
      type: 'session_aborted';
      session_id: string;
      req_id?: string;
    }
  | {
      type: 'upstream_reset';
      session_id?: string;
      commit_seq?: number;
      reason?: string;
      old_synthesis_session_id?: string | null;
      new_synthesis_session_id?: string | null;
    }
  | {
      type: 'upstream_reset_failed';
      session_id?: string;
      commit_seq?: number;
      reason?: string;
      message?: string;
    }
  | {
      type: 'error';
      session_id?: string;
      req_id?: string;
      message: string;
    };
