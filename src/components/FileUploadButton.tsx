import { css } from "@emotion/react";
import { ChangeEvent, ComponentProps, useRef } from "react";
import UploadIcon from "~/assets/svg/upload-icon.svg";
import { Button, FixedButton } from "./Button";

const FileUploadButton = ({
  onFileSelect,
  ...props
}: {
  onFileSelect: (file: File) => void;
} & Omit<ComponentProps<typeof Button>, "onClick">) => {
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const image = event.target.files?.[0];
    if (image) {
      onFileSelect(image);
    }
  };

  return (
    <label css={uploadButtonStyle}>
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        css={css({ display: "none" })}
        accept="image/*"
      />
      <FixedButton
        backgroundColor="#5b52ff"
        leftAddon={<img src={UploadIcon} />}
        onClick={(e) => {
          e.stopPropagation();
          fileInputRef.current?.click(); // input을 강제로 클릭
        }}
        bottomText={
          <>
            powered by
            <span css={css({ color: "#9e77ed", marginLeft: 8 })}>
              Upstage OCR API
            </span>
          </>
        }
        {...props}
      >
        이미지 업로드
      </FixedButton>
    </label>
  );
};

const uploadButtonStyle = css`
  display: flex;
  flex-direction: column;
  align-items: center;
  cursor: pointer;

  .upload-container {
    width: 100%;
    display: flex;
    gap: 10px;
    align-items: center;
    justify-content: center;
    padding: 13px 10px;
    //border: 2px solid #00c853;
    border-radius: 10px;
    background-color: #5b52ff;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    transition: all 0.3s ease-in-out;
  }

  .upload-icon {
    height: 25px;
    margin-left: 5px;
    margin-right: 5px;
  }

  .text-container {
    display: flex;
    flex-direction: column;
  }

  .upload-text {
    font-size: 15px;
    font-weight: 600;
    padding-bottom: 1px;
  }
`;

export default FileUploadButton;
