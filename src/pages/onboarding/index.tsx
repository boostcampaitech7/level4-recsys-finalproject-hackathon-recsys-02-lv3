import { useState, useEffect } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import axios from "axios";
import { css } from "@emotion/react";

import { Tag } from "~/components/OnboardTag";
import { Button } from "~/components/Button";
import { BASE_URL } from "~/libs/api";
import { typedLocalStorage } from "~/utils/localStorage";
import { useUserInfoContext } from "~/utils/userInfoContext";
import { Spacing } from "~/components/Spacing";

export const Component = () => {
  const navigate = useNavigate();
  const { setUserInfo } = useUserInfoContext("onboarding");
  const [searchParams] = useSearchParams();
  const id = searchParams.get("user_id");
  const profileImage = searchParams.get("user_img_url") ?? undefined;

  useEffect(() => {
    console.log(id, profileImage);
    if (id == null) {
      navigate("/");
    } else {
      typedLocalStorage.set("user_id", Number(id));
      typedLocalStorage.set("user_img_url", String(profileImage));
      setUserInfo({ id: Number(id), profileImage });
    }
  }, [id, profileImage]);

  const [selected, setSelected] = useState<string[]>([]);

  const handleTagClick = (tag: string) => {
    setSelected((prevSelected) => {
      if (prevSelected.includes(tag)) {
        return prevSelected.filter((item) => item !== tag);
      } else {
        return [...prevSelected, tag];
      }
    });
  };

  const tags = [
    "열정적인",
    "아날로그 감성",
    "대중적인",
    "독특한",
    "리드미컬한",
    "감성적인",
    "스트리트 감성",
    "신나는",
    "잔잔한",
    "진솔한",
    "영국적인",
    "서정적인",
    "고급스러운",
    "편안한",
    "세련된",
    "영화같은",
    "집중이 잘 되는",
    "거친",
    "트렌디한",
    "스페인풍의",
  ];

  interface Props {
    selected: string[];
    handleTagClick: (tag: string) => void;
  }

  const TagList = ({ selected, handleTagClick }: Props) => {
    return (
      <div css={tagListCSS}>
        {tags.map((tag) => (
          <Tag
            key={tag}
            onClick={() => handleTagClick(`#${tag}`)}
            isSelected={selected.includes(`#${tag}`)}
          >
            #{tag}
          </Tag>
        ))}
      </div>
    );
  };

  const sendData = () => {
    let data = {
      user_id: Number(id),
      tags: selected,
    };
    axios.post(`${BASE_URL}/onboarding`, data, {
      headers: {
        "Content-Type": "application/json",
      },
    });
  };

  return (
    <>
      <Spacing size={8} />
      <TagList selected={selected} handleTagClick={handleTagClick} />

      <Button
        onClick={() => {
          sendData();
          navigate("/home");
        }}
      >
        <div>가게 무드 완성하기</div>
      </Button>
    </>
  );
};

const tagListCSS = css({
  display: "flex",
  flexWrap: "wrap",
  gap: 8,
});
