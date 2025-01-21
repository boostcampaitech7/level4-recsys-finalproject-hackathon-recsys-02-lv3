import { useEffect } from "react";
import { Global, css } from "@emotion/react";
import { QueryClientProvider } from "@tanstack/react-query";
import { ReactQueryDevtools } from "@tanstack/react-query-devtools";
import { RouterProvider } from "react-router-dom";
import { typedLocalStorage } from "~/utils/localStorage";
import { router } from "./router";
import { client } from "~/libs/react-query";

const globalStyles = css`
  .app-wrapper {
    background-color: #121212;
    color: white;
    min-height: 100vh;
    display: flex;
    justify-content: center;
    align-items: center;
  }
`;

function App() {
  useEffect(() => {
    const handleBeforeUnload = () => {
      typedLocalStorage.remove("user_id");
      typedLocalStorage.remove("user_img_url");
    };

    // beforeunload 이벤트 리스너 추가
    window.addEventListener("beforeunload", handleBeforeUnload);

    // 컴포넌트 언마운트 시 이벤트 리스너 제거
    return () => {
      window.removeEventListener("beforeunload", handleBeforeUnload);
    };
  }, []);

  return (
    <QueryClientProvider client={client}>
      <ReactQueryDevtools initialIsOpen={false} />
      <RouterProvider router={router} />
    </QueryClientProvider>
  );
}

export default App;
