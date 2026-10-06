import { viewer, resetLive, pickCell, setLivePlayback } from './original-viewer.js';
import { controlSession, createSession, getBootstrap } from './api';
import type { Frame, MapData, ReplayHeader, SessionStatus } from './types';

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

function message(value: unknown): void {
  element('connection').textContent = value instanceof Error ? value.message : String(value);
}
function status(value: SessionStatus): void {
  session = value;
  setLivePlayback(value.speed, value.paused, ['stopped', 'complete', 'failed'].includes(value.status));
  for (const id of ['urgent', 'block', 'unblock']) element<HTMLButtonElement>(id).disabled = value.status !== 'running';
  pause.disabled = value.status !== 'running';
  pause.textContent = value.paused ? 'Tiếp tục' : 'Tạm dừng';
  element<HTMLButtonElement>('stop').disabled = !['running', 'starting'].includes(value.status);
  element<HTMLButtonElement>('replay').disabled = !['stopped', 'complete', 'failed'].includes(value.status);
  element('modified').hidden = !value.modified;
  message(value.error ?? `${value.method} · ${value.scenario} · ${value.paused ? 'Tạm dừng' : value.status}`);
}
async function control(action: string, data: Record<string, unknown> = {}): Promise<void> {
  try {
    if (!session) throw new Error('Hãy bấm Bắt đầu trước.');
    status(await controlSession(session.id, action, data));
  } catch (error) { message(error); }
}
async function start(): Promise<void> {
  try {
    element('play').hidden = true; element('seek').hidden = true;
    source?.close(); resetLive(); map = undefined; decisions.length = 0;
    status(await createSession(method.value, scenario.value, Number(seed.value)));
    if (!session) return;
    source = new EventSource(`/api/events?session=${encodeURIComponent(session.id)}`);
    source.addEventListener('ready', (event) => {
      const data = JSON.parse((event as MessageEvent<string>).data) as { map: MapData; status: SessionStatus };
      map = data.map; status(data.status);
    });
    source.addEventListener('status', (event) => status(JSON.parse((event as MessageEvent<string>).data) as SessionStatus));
    source.addEventListener('frame', (event) => {
      const frame = JSON.parse((event as MessageEvent<string>).data) as Frame;
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
element('start').onclick = () => void start();
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
    const response = await fetch(session.replay);
    if (!response.ok) throw new Error(`Replay: HTTP ${response.status}`);
    const lines = (await response.text()).trim().split(/\r?\n/);
    const header = JSON.parse(lines.shift() ?? '{}') as ReplayHeader;
    source?.close();
    element('play').hidden = false; element('seek').hidden = false;
    viewer.install({map: header.map, frames: lines.map(line => JSON.parse(line) as Frame), mock: false});
    message('Replay đã nạp. Bấm Phát ở thanh dưới.');
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
  } catch (error) { message(error); }
})();
