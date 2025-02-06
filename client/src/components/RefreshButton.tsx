import { css } from "@emotion/react";
import { ComponentProps } from "react";
import RefreshIcon from "~/assets/svg/ArrowClockwise.svg";

export const RefreshButton = (props: ComponentProps<"button">) => {
  return (
    <button {...props} css={refreshButtonStyle}>
      <img src={RefreshIcon} alt="Refresh Icon" />
      <span css={css({ color: "#e8e8e8", fontSize: 14, marginTop: -2 })}>
        {props.children}
      </span>
    </button>
  );
};

// https://emotion.sh/docs/css-prop#use-the-css-prop
const refreshButtonStyle = css({
  width: "fit-content",
  display: "flex",
  alignItems: "center",
  gap: 6,
  padding: "6px 12px",
  borderRadius: 16,
  height: 30,
  textAlign: "center",
  fontSize: 17,
  fontWeight: 400,
  border: "1px solid #c9c9c9",
  color: "#121212",
});
