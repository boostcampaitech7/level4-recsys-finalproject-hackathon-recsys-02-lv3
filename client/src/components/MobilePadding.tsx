import { css } from "@emotion/react";
import { ReactNode } from "react";
export const MobilePadding = ({ children }: { children: ReactNode }) => {
  return (
    <div
      css={css({
        display: "flex",
        flexDirection: "column",
        width: "100%",
        padding: "0px 20px",
      })}
    >
      {children}
    </div>
  );
};
