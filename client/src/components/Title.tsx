import { css } from "@emotion/react";
import { ReactNode } from "react";

export const Title = ({ children }: { children: ReactNode }) => {
  return <div css={css({ fontSize: 22, fontWeight: 600 })}>{children}</div>;
};
