export interface CommitStageConfig {
  min_chars: number;
  target_chars: number;
  max_chars: number;
  max_wait_ms: number;
  allow_partial_after_ms: number;
}

export interface IntroCacheConfig {
  enabled: boolean;
  text: string;
  max_entries?: number;
  ttl_sec?: number;
  warm_on_session_open?: boolean;
  ignore_errors?: boolean;
  emit_bytes?: number;
  pause_after_ms?: number;
}

export interface ProxyConfig {
  version?: number;
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
    low_latency_first_audio?: boolean;
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
  intro_cache?: IntroCacheConfig;
}

export interface SessionOpenResponse {
  ok: boolean;
  session_id: string;
  config: ProxyConfig;
  ttl_sec: number;
  synthesis_session_id?: string | null;
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
  ignored_chars?: number;
  ignored?: boolean;
  reason?: string;
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
      prefix_cache_crossfade_ms?: number;
      prefix_cache_seam_alignment_enabled?: boolean;
      prefix_cache_seam_search_ms?: number;
      prefix_cache_seam_lookahead_ms?: number;
      prefix_cache_seam_match_ms?: number;
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
      has_prefix_cache?: boolean;
      full_commit_mode?: boolean;
      prefix_cache_preload_context?: boolean;
      prefix_cache_key?: string;
      prefix_cache_key_short?: string;
      prefix_cache_mode?: string;
      prefix_cache_lookup_method?: string;
      full_generation_text_preview?: string;
      full_generation_text_len?: number;
      prefix_cache_text?: string;
      prefix_cache_text_len?: number;
      generation_tail_text_preview?: string;
      generation_tail_text_len?: number;
      prefix_entry_skip_bytes?: number;
      prefix_runtime_skip_adjust_ms?: number;
      planned_prefix_audio_skip_bytes?: number;
      prefix_audio_skip_ms_estimate?: number;
      prefix_cache_held_tail_bytes?: number;
      prefix_cache_crossfade_ms?: number;
      prefix_cache_adaptive_skip_enabled?: boolean;
      prefix_cache_adaptive_search_ms?: number;
      prefix_cache_adaptive_lookahead_ms?: number;
      prefix_cache_adaptive_match_ms?: number;
    }
  | {
      type: 'pcm';
      session_id?: string;
      commit_seq?: number;
      seq: number;
      data: string;
      first_pcm_for_commit?: boolean;
      prefix_cache?: boolean;
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
      has_prefix_cache?: boolean;
      full_commit_mode?: boolean;
      skipped_prefix_pcm_bytes?: number;
      upstream_text_len?: number;
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
      type: 'intro_context_preloaded';
      session_id?: string;
      synthesis_session_id?: string;
      cache_key?: string;
      code_frames?: number;
    }
  | {
      type: 'intro_start';
      session_id?: string;
      req_id?: string;
      cache_key?: string;
      text_preview?: string;
      text_len?: number;
    }
  | {
      type: 'intro_done';
      session_id?: string;
      req_id?: string;
      cache_key?: string;
      pcm_bytes?: number;
    }
  | {
      type: 'intro_error';
      session_id?: string;
      req_id?: string;
      message: string;
    }
  | {
      type: 'prefix_cache_start';
      session_id?: string;
      req_id?: string;
      cache_key?: string;
      cache_key_short?: string;
      text_preview?: string;
      text_len?: number;
      pcm_bytes?: number;
      cache_mode?: string;
      boundary_method?: string;
      planned_held_tail_bytes?: number;
      crossfade_ms?: number;
    }
  | {
      type: 'prefix_cache_done';
      session_id?: string;
      req_id?: string;
      cache_key?: string;
      cache_key_short?: string;
      pcm_bytes?: number;
      held_tail_bytes?: number;
      crossfade_ms?: number;
    }
  | {
      type: 'prefix_cache_generation_skip_done';
      session_id?: string;
      req_id?: string;
      skipped_pcm_bytes?: number;
      skipped_ms_estimate?: number;
      adaptive_skip_method?: string;
      adaptive_skip_score?: number | null;
      adaptive_skip_delta_bytes?: number;
      adaptive_skip_delta_ms?: number;
      adaptive_skip_candidates?: number;
    }
  | {
      type: 'prefix_cache_context_preloaded';
      session_id?: string;
      synthesis_session_id?: string;
      cache_key?: string;
      cache_key_short?: string;
      code_frames?: number;
    }
  | {
      type: 'error';
      session_id?: string;
      req_id?: string;
      message: string;
    };

export interface PrefixCacheItem {
  key: string;
  key_short: string;
  text: string;
  pcm_bytes: number;
  code_frames: number;
  audio_meta?: {
    sample_rate: number;
    channels: number;
    sample_width: number;
  };
  cache_mode: string;
  generation_text?: string;
  lookahead_text?: string | null;
  full_pcm_bytes?: number;
  prefix_audio_skip_bytes?: number;
  prefix_cut_adjust_ms?: number;
  boundary_method?: string;
}

export interface PrefixCacheStatsResponse {
  ok: boolean;
  entries: number;
  max_entries: number;
  items: PrefixCacheItem[];
}

export interface PrefixCacheAddResponse {
  ok: boolean;
  created: PrefixCacheItem[];
  existed: PrefixCacheItem[];
  failed: unknown[];
  created_count: number;
  existed_count: number;
  failed_count: number;
  items: PrefixCacheItem[];
}

export interface PrefixCacheClearResponse {
  ok: boolean;
  message: string;
}
