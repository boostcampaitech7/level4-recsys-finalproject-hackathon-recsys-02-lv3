import { ReactNode } from "react";
import { Navigate } from "react-router-dom";
import useAuthorize from "~/utils/useAuthorize";
import { useUserInfoContext } from "~/utils/userInfoContext";

export const AuthGuard = ({ children }: { children: ReactNode }) => {
  const { id } = useAuthorize();
  return id ? <>{children}</> : <>유저정보 가져오는 중</>;
};
