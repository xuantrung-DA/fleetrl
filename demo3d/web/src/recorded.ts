import { viewer, installReplayWindow } from './original-viewer.js';
import type { RecordingWindow, SavedRecording } from './types';

export function createRecordedPlayer(report: (message: string) => void) {
  let recordingId: string | null = null;
  let generation = 0;
  let loading = false;
  let windowEnd = 0;
  let end = 0;
  let pending: AbortController | null = null;
  let manuallyPaused = true;
  document.getElementById('play')?.addEventListener('click', () => {
    if (recordingId) manuallyPaused = !viewer.getState().playing;
  });

  async function load(time: number, resume: boolean, refresh = false): Promise<void> {
    if (!recordingId) return;
    const ticket = ++generation;
    const id = recordingId;
    pending?.abort();
    pending = new AbortController();
    loading = true;
    if (!refresh) report('Đang mở replay; lần đầu có thể cần giải nén gzip và lập chỉ mục…');
    try {
      const response = await fetch(`/api/recording-window?id=${encodeURIComponent(id)}&start=${time}`, { signal: pending.signal });
      const data = await response.json() as RecordingWindow & { error?: string };
      if (!response.ok) throw new Error(data.error ?? `HTTP ${response.status}`);
      if (ticket !== generation) return;
      const last = data.frames.at(-1);
      if (!last) throw new Error('Replay không có frame.');
      end = data.end_s;
      windowEnd = last.t;
      const state = viewer.getState();
      const target = Math.min(end, Math.max(data.start_s, refresh ? state.time : time));
      installReplayWindow(data, target, refresh ? !manuallyPaused && target < end : resume);
      const modified = document.getElementById('modified');
      if (modified) modified.hidden = !data.frames[0]?.modified;
      report(`REPLAY · ${data.header.session.method} · ${data.header.session.scenario} · seed ${data.header.session.seed} · ${data.total_frames} frame · kết thúc ${end}s`);
    } catch (error) {
      if (ticket === generation) {
        recordingId = null;
        report(error instanceof Error ? error.message : String(error));
      }
    } finally {
      if (ticket === generation) loading = false;
    }
  }

  const timer = window.setInterval(() => {
    const state = viewer.getState();
    if (!recordingId || state.live || loading || !state.playing || windowEnd >= end) return;
    if (windowEnd - state.time < 12) void load(state.time, true, true);
  }, 250);
  window.addEventListener('beforeunload', () => window.clearInterval(timer));

  return {
    async open(id: string): Promise<void> {
      recordingId = id;
      manuallyPaused = true;
      await load(0, false);
    },
    seek(time: number): void {
      if (recordingId) void load(time, viewer.getState().playing);
    },
    cancel(): void {
      generation++;
      pending?.abort();
      recordingId = null;
      loading = false;
    },
  };
}

export async function listRecordings(): Promise<SavedRecording[]> {
  const response = await fetch('/api/recordings');
  if (!response.ok) throw new Error(`Không đọc được danh sách replay: HTTP ${response.status}`);
  return await response.json() as SavedRecording[];
}
