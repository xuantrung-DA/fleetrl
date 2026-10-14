import { viewer, resetLive, pickCell, setLivePlayback } from './original-viewer.js';
import { controlSession, createSession, getBootstrap } from './api';
import type { Frame, MapData, SessionStatus } from './types';
import { createRecordedPlayer, listRecordings } from './recorded';

function element<T extends HTMLElement>(id: string): T {
  const node = document.getElementById(id);
  if (!node) throw new Error(`Missing element: ${id}`);
  return node as T;
}

const panel = document.createElement('section');
panel.innerHTML = `<h2>PHIÊN FLEETRL · LIVE</h2>
<select id="method" aria-label="Phương pháp"></select>
<select id="scenario" aria-label="Kịch bản"></select>
<label>Seed <input id="seed" type="number" value="2000" style="width:80px"></label>
<p><button id="start">Bắt đầu</button> <button id="pause" disabled>Tạm dừng</button>
<button id="stop" disabled>Dừng</button></p>
<p><button id="record">Chạy trước và lưu</button></p>
<small>Chạy hết mô phỏng rồi xem; không huấn luyện lại.</small>
<h3>PHIÊN ĐÃ LƯU</h3>
<select id="saved" aria-label="Bản ghi đã lưu"></select>
<p><button id="refreshSaved">Làm mới</button> <button id="openSaved">Mở replay</button></p>
<p><button id="urgent" disabled>Thêm 5 đơn gấp</button>
<button id="block" disabled>Chặn ô</button> <button id="unblock" disabled>Mở ô</button></p>
<button id="replay" disabled>Xem lại phiên</button>
<p id="connection" role="status">Đang kết nối…</p>
<p id="modified" hidden style="color:#ffc364">Phiên đã can thiệp — KPI chỉ dùng để quan sát.</p>
<small>Ba ứng viên trình diễn từ study14; kết quả phụ thuộc kịch bản.</small>`;
document.querySelector('aside')?.prepend(panel);
const method = element<HTMLSelectElement>('method');
const scenario = element<HTMLSelectElement>('scenario');
const seed = element<HTMLInputElement>('seed');
const pause = element<HTMLButtonElement>('pause');
let session: SessionStatus | null = null;
let source: EventSource | null = null;
let map: MapData | undefined;
let pickMode: 'block' | 'unblock' | null = null;
const decisions: string[] = [];
const recordedPlayer = createRecordedPlayer(message);
const saved = element<HTMLSelectElement>('saved');
async function refreshSaved(): Promise<void> {
  try {
    const previous = saved.value;
    const items = await listRecordings();
    saved.replaceChildren(...items.map(item => new Option(
      `${item.method} · ${item.scenario} · seed ${item.seed} · ${item.status === 'complete' ? 'hoàn tất' : item.status === 'stopped' ? 'đã dừng' : 'bản ghi cũ'} · ${(item.bytes / 1048576).toFixed(1)} MiB · ${item.id}`,
      item.id)));
    if (items.some(item => item.id === previous)) saved.value = previous;
    element<HTMLButtonElement>('openSaved').disabled = items.length === 0;
  } catch (error) { message(error); }
}

function message(value: unknown): void {
  element('connection').textContent = value instanceof Error ? value.message : String(value);
}
function status(value: SessionStatus): void {
  session = value;
  setLivePlayback(value.speed, value.paused, ['stopped', 'complete', 'failed'].includes(value.status));
  for (const id of ['urgent', 'block', 'unblock']) element<HTMLButtonElement>(id).disabled = value.status !== 'running' || !!value.record_only;
  pause.disabled = value.status !== 'running' || !!value.record_only;
  pause.textContent = value.paused ? 'Tiếp tục' : 'Tạm dừng';
  element<HTMLButtonElement>('stop').disabled = !['running', 'starting'].includes(value.status);
  element<HTMLButtonElement>('replay').disabled = !['stopped', 'complete', 'failed'].includes(value.status);
  element('modified').hidden = !value.modified;
  message(value.error ?? `${value.method} · ${value.scenario} · ${value.paused ? 'Tạm dừng' : value.status}`);
  if (value.record_only && value.status === 'running') message(`Đang chạy trước: ${value.method} · ${value.scenario} · ${value.time_s.toFixed(0)}s mô phỏng`);
}
async function control(action: string, data: Record<string, unknown> = {}): Promise<void> {
  try {
    if (!session) throw new Error('Hãy bấm Bắt đầu trước.');
    status(await controlSession(session.id, action, data));
  } catch (error) { message(error); }
}
async function start(recordOnly = false): Promise<void> {
  try {
    element('play').hidden = true; element('seek').hidden = true;
    recordedPlayer.cancel(); source?.close(); resetLive(); map = undefined; decisions.length = 0;
    status(await createSession(method.value, scenario.value, Number(seed.value), recordOnly));
    if (!session) return;
    source = new EventSource(`/api/events?session=${encodeURIComponent(session.id)}`);
    source.addEventListener('ready', (event) => {
      const data = JSON.parse((event as MessageEvent<string>).data) as { map: MapData; status: SessionStatus };
      map = data.map; status(data.status);
    });
    source.addEventListener('status', (event) => {
      const value = JSON.parse((event as MessageEvent<string>).data) as SessionStatus;
      status(value);
      if (['complete', 'stopped'].includes(value.status)) {
        void refreshSaved();
        if (value.record_only && value.status === 'complete') {
          source?.close();
          element('play').hidden = false; element('seek').hidden = false;
          void recordedPlayer.open(value.id);
        }
      }
    });
    source.addEventListener('frame', (event) => {
      const frame = JSON.parse((event as MessageEvent<string>).data) as Frame;
      if (session?.record_only) return;
      viewer.pushLive(frame, map);
      element('modified').hidden = !frame.modified;
      if (frame.decision) {
        decisions.push(`${frame.t}s · ${frame.decision.solver_status} · ${frame.decision.fallback_reason ?? ''}`);
        if (decisions.length > 8) decisions.shift();
      }
      element('log').textContent = decisions.join('\n');
    });
    source.addEventListener('error', (event) => {
      if (event instanceof MessageEvent) message((JSON.parse(event.data as string) as {message: string}).message);
      else message('Mất kết nối với FleetRL. Đang thử kết nối lại…');
    });
  } catch (error) { message(error); }
}
element('start').onclick = () => void start(false);
element('record').onclick = () => void start(true);
element('refreshSaved').onclick = () => void refreshSaved();
element('openSaved').onclick = () => void (async () => {
  try {
    if (!saved.value) throw new Error('Chưa có bản ghi để xem.');
    if (session && ['running', 'starting'].includes(session.status)) status(await controlSession(session.id, 'stop'));
    source?.close(); resetLive();
    element('play').hidden = false; element('seek').hidden = false;
    await recordedPlayer.open(saved.value);
  } catch (error) { message(error); }
})();
element('seek').addEventListener('input', () => {
  if (!viewer.getState().live) recordedPlayer.seek(Number(element<HTMLInputElement>('seek').value));
});
method.onchange = () => { if (session) void start(); };
pause.onclick = () => void control(session?.paused ? 'resume' : 'pause');
element('stop').onclick = () => void control('stop');
element('urgent').onclick = () => void control('urgent_tasks');
for (const action of ['block', 'unblock'] as const) {
  element(action).onclick = () => { pickMode = action; message('Nhấp vào ô cần thao tác trên map.'); };
}
document.querySelector('canvas')?.addEventListener('pointerup', (event) => {
  if (!pickMode) return;
  const cell = pickCell(event);
  if (cell) void control(pickMode, {cell});
  pickMode = null;
});
element<HTMLSelectElement>('speed').onchange = () => {
  if (viewer.getState().live) void control('speed', {speed: Number(element<HTMLSelectElement>('speed').value)});
};
element('replay').onclick = () => void (async () => {
  try {
    if (!session) return;
    source?.close();
    element('play').hidden = false; element('seek').hidden = false;
    await recordedPlayer.open(session.id);
  } catch (error) { message(error); }
})();
void (async () => {
  try {
    const data = await getBootstrap();
    for (const [name, availability] of Object.entries(data.methods)) {
      const option = new Option(name, name);
      option.disabled = !availability.available;
      option.title = availability.reason ?? 'Sẵn sàng';
      method.add(option);
    }
    for (const name of data.scenarios) scenario.add(new Option(name, name));
    scenario.value = data.default_scenario; seed.value = String(data.default_seed);
    message('Chọn phương pháp và kịch bản, rồi bấm Bắt đầu.');
    await refreshSaved();
  } catch (error) { message(error); }
})();
