import { css } from "@emotion/react";
import { ComponentProps, ReactNode } from "react";

export const Description = ({
  children,
  fontSize = 18,
  ...props
}: { children: ReactNode; fontSize?: number } & ComponentProps<"div">) => {
  return (
    <div css={css({ color: "#cacaca", fontSize })} {...props}>
      {children}
    </div>
  );
};
