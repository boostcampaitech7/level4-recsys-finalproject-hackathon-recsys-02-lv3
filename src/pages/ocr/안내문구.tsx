import { css } from "@emotion/react";
import { useMutation } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import OcrExample from "~/assets/ocr-example.png";
import FileUploadButton from "~/components/FileUploadButton";
import { MobilePadding } from "~/components/MobilePadding";
import { Spacing } from "~/components/Spacing";
import { Title } from "~/components/Title";
import { postOcrImageMutation } from "~/remotes";
import { useLoading } from "~/utils/useLoading";
import { useUserId } from "~/utils/userInfoContext";

export const Ocr안내문구 = () => {
  const { mutateAsync: mutateOcrImage } = useMutation(postOcrImageMutation);
  const userId = useUserId();
  const navigate = useNavigate();
  const [isLoading, startLoading] = useLoading();
  const handleSubmit = async (file: File) => {
    const ocrResult = await mutateOcrImage({
      user_id: userId,
      image: file,
    });
    navigate(`/ocr?data=${encodeURIComponent(ocrResult)}`);
  };

  return (
    <MobilePadding>
      <Spacing size={40} />
      <Title>
        사진으로
        <br />
        외부 플레이리스트 가져오기
      </Title>
      <Spacing size={40} />
      <ul
        css={css({
          paddingLeft: 20,
          lineHeight: 1.5,
          fontSize: 18,
          color: "#cacaca",
        })}
      >
        <li>
          앨범커버 등의 부가정보가 포함되지 않도록 상하좌우를 모두 크롭하여
          곡명, 아티스트명만 캡처된 이미지를 사용해주세요.
        </li>
        <br />
        <li> 곡명 - 아티스트명 순서로 정렬된 이미지를 사용해주세요.</li>
      </ul>
      <Spacing size={20} />
      <img
        src={OcrExample}
        css={css({ borderRadius: 20, margin: "0px 10px" })}
      />
      <Spacing size={40} />
      <FileUploadButton
        onFileSelect={(file) => startLoading(handleSubmit(file))}
        loading={isLoading}
      />
      <Spacing size={10} />
    </MobilePadding>
  );
};
