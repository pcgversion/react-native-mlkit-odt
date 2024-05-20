import { NativeModules } from 'react-native';
import type { Frame } from 'react-native-vision-camera';
const { MlkitOdt } = NativeModules;

export type DetectedObjectBounding = {
  originY: number;
  originX: number;
  height: number;
  width: number;
};
export type DetectedObjectLabel = {
  text?: string;
  confidence?: string;
  index?: string;
};
export type ObjectDetectionResult = {
  bounding: DetectedObjectBounding;
  trackingID?: string;
  labels: DetectedObjectLabel[];
};
export type DownloadCustomModelResult = {
  result?: boolean;
};

export enum ObjectDetectorMode {
  STREAM = 0,
  SINGLE_IMAGE = 1,
}
export type ObjectDetectorOptions = {
  detectorMode: ObjectDetectorMode;
  shouldEnableClassification: boolean;
  shouldEnableMultipleObjects: boolean;
  modelName: string;
  customModel: string;
};
const defaultOptions: ObjectDetectorOptions = {
  detectorMode: ObjectDetectorMode.STREAM,
  shouldEnableClassification: false,
  shouldEnableMultipleObjects: false,
  modelName: "",
  customModel: "",
};

const unwrapResult = (res: ObjectDetectionResult | { error: any }) =>
  'error' in res ? Promise.reject(res) : res;

const unwrapResultCustomModelDownload = (res: DownloadCustomModelResult | { error: any }) =>
  'error' in res ? Promise.reject(res) : res;
const wrapper = {
  detectFromUri: (
    uri: string,
    config: ObjectDetectorOptions = defaultOptions
  ): Promise<ObjectDetectionResult[]> =>
    MlkitOdt.detectFromUri(
      uri,
      config.detectorMode === 0 || config.detectorMode === 1
        ? config.detectorMode
        : defaultOptions.detectorMode,
      config.shouldEnableClassification ? 1 : 0,
      config.shouldEnableMultipleObjects ? 1 : 0,
      config.modelName ? config.modelName : "",
      config.customModel ? config.customModel : "",
    ).then(unwrapResult),
};

export const wrapperDownloadCustomModel = {
  downloadCustomModel:(modelName: string): Promise<DownloadCustomModelResult[]>=> 
  MlkitOdt.downloadCustomModel(modelName).then(unwrapResultCustomModelDownload),  
};
type MlkitOdtType = typeof wrapper;

/**
 * Scans barcodes in the passed frame with MLKit
 *
 * @param frame Camera frame
 * @param types Array of barcode types to detect (for optimal performance, use less types)
 * @returns Detected barcodes from MLKit
 */
export function detectObjects(
  frame: Frame,
  options: ObjectDetectorOptions = defaultOptions
): ObjectDetectionResult[] {
  'worklet';
  // @ts-ignore
  // eslint-disable-next-line no-undef
  return __detectObjects(frame, options);
}
export default wrapper as MlkitOdtType;
