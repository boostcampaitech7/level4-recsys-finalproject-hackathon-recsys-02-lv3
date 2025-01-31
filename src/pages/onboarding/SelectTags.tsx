import { css } from "@emotion/react";
import { useState } from "react";
import { Button } from "~/components/Button";
import { Tag } from "~/components/OnboardTag";
import { Spacing } from "~/components/Spacing";

export const SelectTags = ({
  onSubmit,
}: {
  onSubmit: (selectTags: string[]) => Promise<void>;
}) => {
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

  return (
    <>
      <Spacing size={24} />
      <div style={{ textAlign: "center", padding: "15px" }}>
        매장 분위기와 어울리는 태그를 선택해주세요.
      </div>
      <TagList selected={selected} handleTagClick={handleTagClick} />
      <Spacing size={24} />
      <Button css={completeCSS} onClick={() => onSubmit(selected)}>
        다음으로
      </Button>
    </>
  );
};

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
          onClick={() => handleTagClick(`${tag}`)}
          isSelected={selected.includes(`${tag}`)}
        >
          #{tag}
        </Tag>
      ))}
    </div>
  );
};

const tagListCSS = css({
  display: "flex",
  flexWrap: "wrap",
  gap: 8,
  padding: 20,
});
const completeCSS = css({
  width: "calc(100% - 60px)" /* 양쪽에 20px씩 여백 */,
  maxWidth: 600,
  padding: "15px",
  margin: "0 auto",
  display: "flex",
  justifyContent: "center",
  alignItems: "center",
});

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
