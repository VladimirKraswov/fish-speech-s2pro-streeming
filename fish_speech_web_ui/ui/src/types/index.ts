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
}

export type StreamEvent =
  | {
      type: 'session_start';
      session_id: string;
      req_id?: string;
      target_emit_bytes?: number;
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
      text: string;
    }
  | {
      type: 'pcm';
      session_id?: string;
      commit_seq?: number;
      seq: number;
      data: string;
    }
  | {
      type: 'commit_done';
      session_id?: string;
      commit_seq: number;
      upstream_bytes?: number;
    }
  | {
      type: 'session_done';
      session_id: string;
      commit_count?: number;
    }
  | {
      type: 'session_aborted';
      session_id: string;
    }
  | {
      type: 'error';
      session_id?: string;
      message: string;
    };