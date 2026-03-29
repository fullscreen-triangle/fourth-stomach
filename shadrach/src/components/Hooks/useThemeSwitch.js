import { useEffect } from "react";

export function useThemeSwitch() {
  useEffect(() => {
    document.documentElement.classList.add("dark");
  }, []);

  return ["dark", () => {}];
}
