import { css } from "@emotion/react";
import { ReactNode } from "react";

export const Title = ({
  children,
  fontSize = 22,
  color = "#ffffff",
}: {
  children: ReactNode;
  fontSize?: number;
  color?: string;
}) => {
  return <div css={css({ color, fontSize, fontWeight: 600 })}>{children}</div>;
};
