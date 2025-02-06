/* eslint-disable @typescript-eslint/no-explicit-any */
export * from "@suspensive/react-query";

import {
  QueryKey,
  UseMutationOptions,
  UseQueryOptions,
  WithRequired,
} from "@tanstack/react-query";

/**
 * 주어진 query에 select option을 붙여 줍니다
 * @example
 * ```typescript
 * const postsCountQuery = select(postsQuery, list => list.length)
 *
 * useSuspenseQuery(postsQuery).data      // Post[]
 * useSuspenseQuery(postsCountQuery).data // number
 * ```
 */
export const select = <
  TQueryFnData = unknown,
  TError = unknown,
  TData = TQueryFnData,
  TQueryKey extends QueryKey = QueryKey
>(
  options: WithRequired<
    UseQueryOptions<TQueryFnData, TError, any, TQueryKey>,
    "queryKey" | "queryFn"
  >,
  selector: (data: TQueryFnData) => TData
) => ({ ...options, select: selector });

/**
 * mutation을 수행하기 위한 옵션 설정 함수입니다.
 */
export const mutationOptions = <
  TData = unknown,
  TError = unknown,
  TVariables = void,
  TContext = unknown
>(
  options: UseMutationOptions<TData, TError, TVariables, TContext>
) => {
  return options;
};
