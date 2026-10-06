import type { Bootstrap, SessionStatus } from "./types";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  const body = (await response.json()) as T & { error?: string };
  if (!response.ok) {
    throw new Error(body.error ?? `HTTP ${response.status}`);
  }
  return body;
}

export const getBootstrap = (): Promise<Bootstrap> => request<Bootstrap>("/api/bootstrap");

export const createSession = (
  method: string,
  scenario: string,
  seed: number,
): Promise<SessionStatus> =>
  request<SessionStatus>("/api/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ method, scenario, seed }),
  });

export const controlSession = (
  session: string,
  action: string,
  data: Record<string, unknown> = {},
): Promise<SessionStatus> =>
  request<SessionStatus>("/api/control", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session, action, ...data }),
  });
