import { css } from "@emotion/react";
import { ComponentProps } from "react";

interface ButtonColorProps {
  backgroundColor?: string;
  color?: string;
}

export const Button = ({
  backgroundColor = "#1ED760",
  color = "#fff",
  ...props
}: ComponentProps<"button"> & ButtonColorProps) => {
  return <button {...props} css={buttonStyle({ backgroundColor, color })} />;
};

// https://emotion.sh/docs/css-prop#use-the-css-prop
const buttonStyle = ({ backgroundColor, color }: ButtonColorProps) =>
  css({
    width: "fit-content", // 혹은 "max-content" / 고정 px 등
    padding: "0 100px", // 버튼 내부 여백
    margin: "0 auto", // 수평 중앙 정렬
    borderRadius: 8,
    height: 56,
    backgroundColor: backgroundColor,
    color: color,
    display: "flex",
    justifyContent: "center",
    alignItems: "center",

    "&:disabled": {
      backgroundColor: "#656565",
    },
  });
