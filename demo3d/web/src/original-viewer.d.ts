import type { Frame, MapData, RecordingWindow } from './types';
export const viewer: {
  pushLive(frame: Frame, map?: MapData): void;
  install(data: { map: MapData; frames: Frame[]; mock: boolean }): void;
  getState(): { live: boolean; time: number; playing: boolean };
};
export function resetLive(): void;
export function setLivePlayback(speed: number, paused: boolean, finished: boolean): void;
export function installReplayWindow(data: RecordingWindow, target: number, resume: boolean): void;
export function pickCell(event: PointerEvent): [number, number] | null;
