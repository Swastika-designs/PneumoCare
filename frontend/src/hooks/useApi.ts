import { useCallback, useEffect, useRef, useState } from "react";
import { describeError, type ApiErrorInfo } from "../services/api";

export interface AsyncState<T> {
  data: T | null;
  error: ApiErrorInfo | null;
  loading: boolean;
}

/**
 * Run an async action on demand (e.g. a "Run analysis" button). The latest `fn` is kept in a
 * ref so `run` is stable even when callers pass inline arrow functions. Responses from older
 * calls are ignored so fast clicking never shows mismatched results.
 */
export function useAction<T, A extends unknown[] = []>(fn: (...args: A) => Promise<T>) {
  const [state, setState] = useState<AsyncState<T>>({ data: null, error: null, loading: false });
  const fnRef = useRef(fn);
  fnRef.current = fn;
  const callId = useRef(0);
  const run = useCallback(async (...args: A): Promise<T | null> => {
    const id = ++callId.current;
    setState((s) => ({ ...s, loading: true, error: null }));
    try {
      const data = await fnRef.current(...args);
      if (id === callId.current) setState({ data, error: null, loading: false });
      return data;
    } catch (err) {
      if (id === callId.current) setState((s) => ({ ...s, error: describeError(err), loading: false }));
      return null;
    }
  }, []);
  const reset = useCallback(() => setState({ data: null, error: null, loading: false }), []);
  return { ...state, run, reset };
}

/**
 * Fetch automatically on mount and whenever the serialised dependency key changes.
 * React StrictMode (development) runs effects twice on mount; the ref makes sure the same key
 * is requested only once, so backend analysis runs are not recorded twice.
 */
export function useFetch<T>(fn: () => Promise<T>, key: unknown[] = []) {
  const action = useAction(fn);
  const { run } = action;
  const serialized = JSON.stringify(key);
  const lastKey = useRef<string | null>(null);
  useEffect(() => {
    if (lastKey.current === serialized) return;
    lastKey.current = serialized;
    void run();
  }, [run, serialized]);
  const retry = useCallback(() => run(), [run]);
  return { ...action, run: retry };
}
