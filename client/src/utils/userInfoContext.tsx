import { createContext } from "~/libs/context";

export interface UserInfo {
  id?: number;
  profileImage?: string;
}

export const [UserInfoProvider, useUserInfoContext] = createContext<{
  userInfo: UserInfo;
  setUserInfo: (value: UserInfo) => void;
}>("UserInfo");

export const useUserId = () => {
  const { userInfo } = useUserInfoContext("useUser");
  if (userInfo.id == null) {
    throw new Error();
  }
  return Number(userInfo.id);
};
