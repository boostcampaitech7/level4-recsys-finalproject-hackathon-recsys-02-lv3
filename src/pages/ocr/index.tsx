import { AuthGuard } from "~/components/AuthGuard";
import { Ocr안내문구 } from "./안내문구";
import { Ocr트랙리스트 } from "./트랙리스트";
import { useSearchParams } from "react-router-dom";
import { OcrTrackRequest } from "~/remotes/dio";

const OcrPage = () => {
  const [searchParams] = useSearchParams();
  const serializedData = searchParams.get("data");
  const ocrTracks: OcrTrackRequest[] = serializedData
    ? JSON.parse(decodeURIComponent(serializedData))
    : [];

  return (
    <>
      {ocrTracks.length > 0 ? (
        <Ocr트랙리스트 ocrTracks={ocrTracks} />
      ) : (
        <Ocr안내문구 />
      )}
    </>
  );
};

export const Component = () => (
  <AuthGuard>
    <OcrPage />
  </AuthGuard>
);
