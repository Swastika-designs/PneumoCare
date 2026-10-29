import { useSearchParams } from "react-router-dom";

/** Keep the active tab in the URL (?tab=...) so tutorial links can open a specific workspace. */
export function useTab<T extends string>(allowed: readonly T[], fallback: T): [T, (t: T) => void] {
  const [params, setParams] = useSearchParams();
  const current = params.get("tab") as T | null;
  const tab = current && allowed.includes(current) ? current : fallback;
  const setTab = (t: T) => {
    const next = new URLSearchParams(params);
    next.set("tab", t);
    setParams(next, { replace: true });
  };
  return [tab, setTab];
}
