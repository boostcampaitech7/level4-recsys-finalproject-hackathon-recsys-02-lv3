import { css } from "@emotion/react";
import { useQuery } from "@tanstack/react-query";
import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import FileUploadButton from "~/components/FileUploadButton";
import { ProfileImage } from "~/components/ProfileImage";
import { Spacing } from "~/components/Spacing";
import { playlistQuery } from "~/remotes";
import useAuthorize from "~/utils/useAuthorize";
import { useUserInfoContext } from "~/utils/userInfoContext";

export const Component = () => {
  const navigate = useNavigate();
  useAuthorize();
  const { userInfo } = useUserInfoContext("useAuthorize");
  const { id } = userInfo;

  const { data } = useQuery({
    ...playlistQuery(Number(id)),
    enabled: id != null,
  });

  useEffect(() => {
    if (id == null) {
      navigate("/");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  return (
    <>
      <div>
        <Spacing size={10} />
        <ProfileImage />
      </div>
      <Spacing size={10} />
      <div
        css={css({
          display: "flex",
          marginLeft: 15,
          gap: 30,
          width: "100%",
          overflowX: "auto",
        })}
      >
        {data?.items.map((v) => (
          <div
            key={v.playlist_id}
            onClick={() => navigate(`/playlist/${v.playlist_id}`)}
          >
            <div
              css={css({
                width: "150px", // 원하는 크기로 조정 가능
                height: "150px",
                overflow: "hidden",
                display: "flex",
                cursor: "pointer",
                alignItems: "center",
                justifyContent: "center",
              })}
            >
              <img
                src={v.playlist_img_url}
                css={css({
                  width: "100%",
                  height: "100%",
                  objectFit: "cover",
                  objectPosition: "center",
                })}
              />
            </div>
            {v.playlist_name}
          </div>
        ))}
      </div>
      <br />
      <div>
        <FileUploadButton onFileSelect={(file) => console.log(file.name)} />
      </div>
    </>
  );
};
