import { useSuspenseQuery } from "@suspensive/react-query";
import { ReactNode, Suspense } from "react";
import { Navigate } from "react-router-dom";
import { userEmbeddingQuery } from "~/remotes";
import useAuthorize from "~/utils/useAuthorize";
import { FullScreenLoader } from "./FullScreenLoader";

export const AuthGuard = ({ children }: { children: ReactNode }) => {
  const { id } = useAuthorize();

  if (location.pathname === "/") {
    return <>{children}</>;
  }
  return id ? (
    <Suspense fallback={<FullScreenLoader />}>
      <AssertEmbedded id={id}>{children}</AssertEmbedded>
    </Suspense>
  ) : (
    <FullScreenLoader />
  );
};

const AssertEmbedded = ({
  id,
  children,
}: {
  id: number;
  children: ReactNode;
}) => {
  const { data } = useSuspenseQuery(userEmbeddingQuery(id));

  if (data.exist === false && location.pathname !== "/onboarding") {
    return <Navigate to={"/onboarding"} />;
  }
  return <>{children}</>;
};
