import {
  createContext as createContextRaw,
  PropsWithChildren,
  useContext as useContextRaw,
  useMemo,
} from "react";

export class ContextError extends Error {
  static ErrorName = "InvalidContextUsage";
  static isContextError(error: Error) {
    return error.name === ContextError.ErrorName;
  }

  constructor(message: string) {
    super(message);
    this.name = ContextError.ErrorName;
  }
}

export function createContext<DefaultContextType extends object | null>(
  rootComponentName: string,
  defaultContext?: DefaultContextType
) {
  const Context = createContextRaw<DefaultContextType | undefined>(
    defaultContext
  );

  function Provider<ContextValueType extends DefaultContextType>(
    props: PropsWithChildren<ContextValueType>
  ) {
    const { children, ...context } = props;

    const value = useMemo(() => context, [context]) as ContextValueType;

    return <Context.Provider value={value}>{children}</Context.Provider>;
  }

  function useContext<ContextValueType extends DefaultContextType>(
    consumerName: string
  ) {
    const context = useContextRaw(Context) as ContextValueType | undefined;
    if (context == null) {
      throw new ContextError(
        `${consumerName}은 ${rootComponentName} 하위에서 사용해야 합니다.`
      );
    }

    return context;
  }

  Provider.displayName = `${rootComponentName}Provider`;
  return [Provider, useContext] as const;
}
