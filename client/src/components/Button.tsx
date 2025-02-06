import { css } from "@emotion/react";
import Lottie from "lottie-react";
import { ComponentProps, ReactElement, ReactNode } from "react";
import loadingLottie from "~/assets/loading-small-lottie.json";
import { rgba } from "emotion-rgba";
import { Spacing } from "./Spacing";
interface ButtonColorProps {
  backgroundColor?: string;
  color?: string;
  loading?: boolean;
  leftAddon?: ReactElement;
  bottomText?: ReactNode;
}

export const Button = ({
  backgroundColor = "#1ED760",
  color = "#fff",
  loading,
  leftAddon,
  bottomText,
  ...props
}: ComponentProps<"button"> & ButtonColorProps) => {
  return (
    <>
      <button
        {...props}
        css={buttonStyle({ backgroundColor, color })}
        disabled={loading || props.disabled}
      >
        {loading ? (
          <Lottie
            animationData={loadingLottie}
            loop={true}
            css={css({ height: 120, width: 120 })}
          />
        ) : (
          <>
            {leftAddon}
            {props.children}
          </>
        )}
      </button>
      {bottomText && (
        <div
          css={css({
            margin: "10px auto",
            color: "#818181",
            textAlign: "center",
          })}
        >
          {bottomText}
        </div>
      )}
    </>
  );
};

const buttonStyle = ({
  backgroundColor = "#1ED760",
  color,
}: ButtonColorProps) =>
  css({
    width: "100%",
    margin: "0 auto", // 수평 중앙 정렬
    borderRadius: 8,
    height: 56,
    backgroundColor: backgroundColor,
    color: color,
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    gap: 10,

    "&:disabled": {
      backgroundColor: rgba(backgroundColor, 0.6),
      color: "#b2b2b2",
    },

    "&:active": {
      backgroundColor: rgba(backgroundColor, 0.9),
    },
  });

export const FixedButton = (props: ComponentProps<typeof Button>) => {
  return (
    <>
      <Spacing size={props.bottomText ? 84 : 60} />
      <div
        css={css({
          bottom: 0,
          width: "calc(100% - 40px)",
          maxWidth: 710,
          position: "fixed",
        })}
      >
        <div
          css={css({
            width: "100%",
            height: 40,
            background: "linear-gradient(#12121200, #121212)",
          })}
        />
        <div
          css={css({
            paddingBottom: 20,
            backgroundColor: "#121212",
          })}
        >
          <Button {...props} />
        </div>
      </div>
    </>
  );
};
