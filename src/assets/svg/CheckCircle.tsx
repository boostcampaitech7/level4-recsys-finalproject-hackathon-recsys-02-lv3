export const CheckCircle = ({ color = "#1ED760" }: { color?: string }) => {
  return (
    <svg
      width="25"
      height="25"
      viewBox="0 0 30 31"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <circle cx="15" cy="15.6768" r="15" fill={color} />
      <path
        d="M5.90918 15.6768L11.9701 21.7377L24.0906 9.61584"
        stroke="white"
        strokeWidth="3"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
};
