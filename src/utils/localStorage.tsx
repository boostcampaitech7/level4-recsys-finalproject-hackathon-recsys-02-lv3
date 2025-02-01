// 로컬 스토리지 유틸리티 함수
export type KeysProps = {
  [key: string]: string;
};

export const typedLocalStorage = {
  get<T>(key: string): T | undefined {
    const result = localStorage.getItem(key);
    return (result as T) ?? undefined;
  },

  set<T>(key: string, value: T | undefined): void {
    localStorage.setItem(key, JSON.stringify(value));
  },

  remove(key: string): void {
    localStorage.removeItem(key);
  },
};
