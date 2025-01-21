// 로컬 스토리지 유틸리티 함수
export type KeysProps = {
  [key: string]: any;
};

export const typedLocalStorage = {
  get<T = any>(key: string): T | undefined {
    const result = localStorage.getItem(key);
    if (result === null) {
      return undefined;
    }

    try {
      return JSON.parse(result) as T;
    } catch (error) {
      console.error(`Error parsing localStorage key "${key}":`, error);
      return undefined;
    }
  },

  set<T = any>(key: string, value: T | undefined): void {
    if (value === null) {
      localStorage.removeItem(key);
    } else {
      localStorage.setItem(key, JSON.stringify(value));
    }
  },

  remove(key: string): void {
    localStorage.removeItem(key);
  },
};

// export const typedLocalStorage = {
//   get<K extends keyof KeysProps>(key: K): KeysProps[K] | null {
//     const result = localStorage.getItem(key);
//     if (result === null) {
//       return null;
//     }

//     return result;
//   },

//   set<K extends keyof KeysProps>(key: K, value: KeysProps[K] | null) {
//     if (value === null) {
//       return localStorage.removeItem(key);
//     } else {
//       return localStorage.setItem(key, JSON.stringify(value));
//     }
//   },

//   remove<K extends keyof KeysProps>(key: K) {
//     return localStorage.removeItem(key);
//   },
// };
