import { ReactNode } from "react";
import useAuthorize from "~/utils/useAuthorize";

export const AuthGuard = ({ children }: { children: ReactNode }) => {
  const { id } = useAuthorize();
  return id ? <>{children}</> : <>유저정보 가져오는 중</>;
};
