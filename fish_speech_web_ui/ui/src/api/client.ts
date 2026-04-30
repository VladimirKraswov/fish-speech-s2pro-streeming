import type {
  SessionOpenResponse,
  SessionAppendResponse,
  PrefixCacheStatsResponse,
  PrefixCacheAddResponse,
  PrefixCacheClearResponse,
} from '../types';

async function readError(resp: Response): Promise<string> {
  const payload = await resp.json().catch(() => null);
  if (payload?.detail) return String(payload.detail);
  if (payload?.message) return String(payload.message);
  return `${resp.status} ${resp.statusText}`;
}

export class FishProxyClient {
  private baseUrl: string;

  constructor(baseUrl = '') {
    this.baseUrl = baseUrl.replace(/\/$/, '');
  }

  async getHealth() {
    const resp = await fetch(`${this.baseUrl}/health`, { cache: 'no-store' });
    if (!resp.ok) throw new Error(await readError(resp));
    return resp.json();
  }

  async openSession(configText: string): Promise<SessionOpenResponse> {
    const resp = await fetch(`${this.baseUrl}/session/open`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ config_text: configText }),
    });
    if (!resp.ok) throw new Error(await readError(resp));
    return resp.json();
  }

  async appendText(
    sessionId: string,
    text: string,
    options?: { cache?: string; cache_key?: string }
  ): Promise<SessionAppendResponse> {
    const body: Record<string, unknown> = { text };
    if (options?.cache_key) body.cache_key = options.cache_key;
    else if (options?.cache) body.cache = options.cache;

    const resp = await fetch(`${this.baseUrl}/session/${sessionId}/append`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!resp.ok) throw new Error(await readError(resp));
    return resp.json();
  }

  async flushSession(sessionId: string, reason = 'manual_flush'): Promise<SessionAppendResponse> {
    const resp = await fetch(`${this.baseUrl}/session/${sessionId}/flush`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ reason }),
    });
    if (!resp.ok) throw new Error(await readError(resp));
    return resp.json();
  }

  async finishSession(sessionId: string, reason = 'input_finished'): Promise<SessionAppendResponse> {
    const resp = await fetch(`${this.baseUrl}/session/${sessionId}/finish`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ reason }),
    });
    if (!resp.ok) throw new Error(await readError(resp));
    return resp.json();
  }

  async closeSession(sessionId: string): Promise<{ ok: boolean; closed: boolean }> {
    const resp = await fetch(`${this.baseUrl}/session/close`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId }),
    });
    if (!resp.ok) throw new Error(await readError(resp));
    return resp.json();
  }

  getStreamUrl(sessionId: string): string {
    return `${this.baseUrl}/session/${sessionId}/pcm-stream`;
  }

  async getPrefixCacheStats(): Promise<PrefixCacheStatsResponse> {
    const resp = await fetch(`${this.baseUrl}/prefix-cache/stats`, {
      cache: 'no-store',
    });
    if (!resp.ok) throw new Error(await readError(resp));
    return resp.json();
  }

  async addPrefixCache(
    configText: string,
    text: string
  ): Promise<PrefixCacheAddResponse> {
    const resp = await fetch(`${this.baseUrl}/prefix-cache/add`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ config_text: configText, text }),
    });
    if (!resp.ok) throw new Error(await readError(resp));
    return resp.json();
  }

  async clearPrefixCache(): Promise<PrefixCacheClearResponse> {
    const resp = await fetch(`${this.baseUrl}/prefix-cache/clear`, {
      method: 'POST',
    });
    if (!resp.ok) throw new Error(await readError(resp));
    return resp.json();
  }
}