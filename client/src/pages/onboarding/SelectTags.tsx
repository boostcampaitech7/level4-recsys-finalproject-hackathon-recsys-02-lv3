import { useState } from "react";
import { FixedButton } from "~/components/Button";
import { MobilePadding } from "~/components/MobilePadding";
import { Tag } from "~/components/OnboardTag";
import { Spacing } from "~/components/Spacing";
import { Title } from "~/components/Title";
import { useLoading } from "~/utils/useLoading";

export const SelectTags = ({
  onSubmit,
}: {
  onSubmit: (selectTags: number[]) => Promise<void>;
}) => {
  const [isLoading, startLoading] = useLoading();
  const [selected, setSelected] = useState<number[]>([]);

  const handleTagClick = (tag: number) => {
    setSelected((prevSelected) => {
      if (prevSelected.includes(tag)) {
        return prevSelected.filter((item) => item !== tag);
      } else {
        return [...prevSelected, tag];
      }
    });
  };

  return (
    <MobilePadding>
      <Spacing size={24} />
      <Title>
        매장 분위기와 어울리는
        <br />
        태그를 선택해주세요
      </Title>
      <Spacing size={40} />
      <TagList selected={selected} handleTagClick={handleTagClick} />
      <Spacing size={40} />
      <FixedButton
        onClick={() => startLoading(onSubmit(selected))}
        loading={isLoading}
        disabled={selected.length === 0}
      >
        다음으로
      </FixedButton>
    </MobilePadding>
  );
};

interface Props {
  selected: number[];
  handleTagClick: (tag: number) => void;
}

const TagList = ({ selected, handleTagClick }: Props) => {
  return (
    <div>
      {tags.map((tag) => (
        <Tag
          key={tag.tag_id}
          onClick={() => handleTagClick(tag.tag_id)}
          isSelected={selected.includes(tag.tag_id)}
        >
          #{tag.tag}
        </Tag>
      ))}
    </div>
  );
};

type TagType = {
  tag_id: number;
  tag: string;
};

const tags: TagType[] = [
  {
    tag_id: 1,
    tag: "록스피릿",
  },
  {
    tag_id: 2,
    tag: "신나는",
  },
  {
    tag_id: 3,
    tag: "대중적인",
  },
  {
    tag_id: 4,
    tag: "힙한",
  },
  {
    tag_id: 5,
    tag: "재즈소울",
  },
  {
    tag_id: 6,
    tag: "헤비메탈",
  },
  {
    tag_id: 7,
    tag: "어쿠스틱한",
  },
  {
    tag_id: 8,
    tag: "반항적인",
  },
  {
    tag_id: 9,
    tag: "인디",
  },
  {
    tag_id: 10,
    tag: "몽환적인",
  },
  {
    tag_id: 11,
    tag: "정열적인",
  },
  {
    tag_id: 12,
    tag: "레트로한",
  },
  {
    tag_id: 13,
    tag: "알앤비",
  },
  {
    tag_id: 14,
    tag: "집중",
  },
  {
    tag_id: 15,
    tag: "chill한",
  },
  {
    tag_id: 16,
    tag: "청량한",
  },
  {
    tag_id: 17,
    tag: "아방가르드한",
  },
  {
    tag_id: 18,
    tag: "감성적인",
  },
];
