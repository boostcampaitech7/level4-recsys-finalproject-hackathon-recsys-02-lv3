import { css } from "@emotion/react";
import { ChangeEvent } from "react";
import UploadIcon from "~/assets/svg/upload-icon.svg";

const FileUploadButton = ({
  onFileSelect,
}: {
  onFileSelect: (file: File) => void;
}) => {
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
        onChange={handleFileChange}
        style={{ display: "none" }}
      />
      <div className="upload-container">
        <img src={UploadIcon} alt="Upload Icon" className="upload-icon" />
        <div
          css={css({
            display: "flex",
            textAlign: "center",
            flexDirection: "column",
            alignItems: "center",
            marginLeft: 15,
          })}
        >
          <span className="upload-text">
            사진으로 외부 플레이리스트 불러오기
          </span>
          <span className="upload-subtext">Upstage OCR API</span>
        </div>
      </div>
    </label>
  );
};

const uploadButtonStyle = css`
  display: inline-block;
  cursor: pointer;

  .upload-container {
    display: flex;
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    align-items: center;
    justify-content: flex-start;
    width: 310px;
    padding: 13px 10px;
    border: 2px solid #00c853;
    border-radius: 10px;
    background-color: #fff;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    transition: all 0.3s ease-in-out;

    &:hover {
      background-color: #f0f0f0;
    }
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
    font-weight: bold;
    padding-bottom: 1px;
    color: #000;
  }

  .upload-subtext {
    font-size: 10px;
    color: #777;
  }
`;

export default FileUploadButton;
