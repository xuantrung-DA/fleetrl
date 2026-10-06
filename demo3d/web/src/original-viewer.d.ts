import type { Frame, MapData } from './types';
export const viewer: {
  pushLive(frame: Frame, map?: MapData): void;
  install(data: { map: MapData; frames: Frame[]; mock: boolean }): void;
  getState(): { live: boolean };
};
export function resetLive(): void;
export function setLivePlayback(speed: number, paused: boolean, finished: boolean): void;
export function pickCell(event: PointerEvent): [number, number] | null;
