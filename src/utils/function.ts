const NamedFunctionRegEx = /(?<=^function\s+)(\w+)\s*(?=\()/;
const NamedMethodRegEx = /^(\w+)\s*(?=\()/;
const AnonymousFunctionRegEx = /(?<=^function\s*)()\s*(?=\()/;

const BodyThisProperty = /(this(\.\w+)+)\b(?!\s*\()/g;

export function getGpuActivationFunction(thisObj: any, fn: (x: number) => number, name: string): string {
    const fnStr = transformFunctionMemberExpr(thisObj, fn.toString());

    if (fnStr.match(NamedFunctionRegEx)) {
        return fnStr.replace(NamedFunctionRegEx, name);
    } else if (fnStr.match(NamedMethodRegEx)) {
        return "function " + fnStr.replace(NamedMethodRegEx, name);
    } else if (fnStr.match(AnonymousFunctionRegEx)) {
        return fnStr.replace(AnonymousFunctionRegEx, ` ${name}`);
    }

    throw new Error("Unsupported function");
}

export function transformFunctionMemberExpr(thisObj: any, fnStr: string): string {
    const matches = fnStr.matchAll(BodyThisProperty);
    for (const [match] of matches) {
        const path = match.split(".").slice(1);
        fnStr = fnStr.replace(match, _getPropertyValue(thisObj, path));
    }

    return fnStr;
}

function _getPropertyValue(obj: any, path: string[]) {
    let ret = obj;
    for (const part of path) {
        if (ret === undefined) throw new Error(`Object has not any value at path : ${part}`)

        ret = ret[part];
    }

    return ret;
}