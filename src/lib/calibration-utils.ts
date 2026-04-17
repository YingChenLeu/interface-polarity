import { 
  FittsTrialData, 
  HicksTrialData, 
  FittsLawEquation, 
  HicksLawEquation,
  TaskType 
} from "@/types/study";

const FALLBACK_FITTS: FittsLawEquation = { a: 200, b: 150, r2: 0 };
const FALLBACK_HICKS: HicksLawEquation = { a: 200, b: 150, r2: 0 };

// Calculate Index of Difficulty for Fitts' Law: ID = log2(D/W + 1)
export function calculateIndexOfDifficulty(distance: number, width: number): number {
  return Math.log2(distance / width + 1);
}

// Linear regression helper
function linearRegression(xValues: number[], yValues: number[]): { a: number; b: number; r2: number } {
  const n = xValues.length;
  if (n < 2) {
    return { a: 0, b: 0, r2: 0 };
  }

  const sumX = xValues.reduce((acc, x) => acc + x, 0);
  const sumY = yValues.reduce((acc, y) => acc + y, 0);
  const sumXY = xValues.reduce((acc, x, i) => acc + x * yValues[i], 0);
  const sumX2 = xValues.reduce((acc, x) => acc + x * x, 0);
  const sumY2 = yValues.reduce((acc, y) => acc + y * y, 0);

  const denominator = n * sumX2 - sumX * sumX;
  if (denominator === 0) {
    return { a: sumY / n, b: 0, r2: 0 };
  }

  const b = (n * sumXY - sumX * sumY) / denominator;
  const a = (sumY - b * sumX) / n;

  // Calculate R² (coefficient of determination)
  const yMean = sumY / n;
  const ssTotal = yValues.reduce((acc, y) => acc + Math.pow(y - yMean, 2), 0);
  const ssResidual = yValues.reduce((acc, y, i) => {
    const predicted = a + b * xValues[i];
    return acc + Math.pow(y - predicted, 2);
  }, 0);
  const r2 = ssTotal > 0 ? 1 - ssResidual / ssTotal : 0;

  return { a, b, r2: Math.max(0, r2) };
}

// Compute Fitts' Law equation from trial data
// MT = a + b * ID where ID = log2(D/W + 1)
// Uses per-condition mean MT (like Hicks) to reduce trial-level noise and improve R².
export function computeFittsEquation(trials: FittsTrialData[]): FittsLawEquation {
  const totalTrials = trials.length;
  const successfulTrials = trials.filter(t => t.success);
  const excludedFailed = totalTrials - successfulTrials.length;

  if (successfulTrials.length < 2) {
    return {
      ...FALLBACK_FITTS,
      stats: {
        totalTrials,
        excludedFailed,
        excludedTooFast: 0,
        includedTrials: successfulTrials.length,
      },
    };
  }

  // Drop very short movement times (anticipatory or misclicks)
  const MIN_MT_MS = 100;
  const filteredTrials = successfulTrials.filter(
    (t) => t.movementTimeMs >= MIN_MT_MS
  );
  const excludedTooFast = successfulTrials.length - filteredTrials.length;

  if (filteredTrials.length < 2) {
    return {
      ...FALLBACK_FITTS,
      stats: {
        totalTrials,
        excludedFailed,
        excludedTooFast,
        includedTrials: filteredTrials.length,
      },
    };
  }

  // Aggregate to mean MT per (width, distance) condition. This reduces trial-level
  // noise and yields a clearer Fitts relationship and higher R².
  const mtByCondition = new Map<string, number[]>();
  for (const t of filteredTrials) {
    const key = `${t.targetWidth}-${t.targetDistance}`;
    const list = mtByCondition.get(key) ?? [];
    list.push(t.movementTimeMs);
    mtByCondition.set(key, list);
  }

  const xValues: number[] = [];
  const yValues: number[] = [];
  for (const [key, mts] of mtByCondition) {
    const [w, d] = key.split("-").map(Number);
    const id = calculateIndexOfDifficulty(d, w);
    const meanMt = mts.reduce((acc, v) => acc + v, 0) / mts.length;
    xValues.push(id);
    yValues.push(meanMt);
  }

  // Need at least 2 distinct conditions to fit a line; 3+ is better for R²
  if (xValues.length < 2) {
    return {
      ...FALLBACK_FITTS,
      stats: {
        totalTrials,
        excludedFailed,
        excludedTooFast,
        includedTrials: filteredTrials.length,
      },
    };
  }

  const equation = linearRegression(xValues, yValues);
  return {
    ...equation,
    stats: {
      totalTrials,
      excludedFailed,
      excludedTooFast,
      includedTrials: filteredTrials.length,
    },
  };
}

// Compute Hick's Law equation from trial data
// RT = a + b * log2(n) where n is number of choices
export function computeHicksEquation(trials: HicksTrialData[]): HicksLawEquation {
  const totalTrials = trials.length;
  // 1) Keep only correct trials
  const correctTrials = trials.filter(t => t.correct);
  const excludedIncorrect = totalTrials - correctTrials.length;

  // 2) Remove extreme outliers (very fast or very slow responses),
  //    which add noise unrelated to log2(n) and hurt R².
  //    Thresholds are in milliseconds.
  const filteredTrials = correctTrials.filter(t => 
    t.reactionTimeMs >= 150 && t.reactionTimeMs <= 2500
  );
  const excludedOutOfRange = correctTrials.length - filteredTrials.length;

  if (filteredTrials.length < 3) {
    return {
      ...FALLBACK_HICKS,
      stats: {
        totalTrials,
        excludedIncorrect,
        excludedOutOfRange,
        includedTrials: filteredTrials.length,
      },
    };
  }

  // 3) Aggregate to mean RT per choice level. This reduces trial-level noise
  //    and typically yields a clearer Hick's Law relationship and higher R².
  const rtByChoices = new Map<number, number[]>();
  for (const t of filteredTrials) {
    const list = rtByChoices.get(t.numChoices) ?? [];
    list.push(t.reactionTimeMs);
    rtByChoices.set(t.numChoices, list);
  }

  const uniqueChoiceLevels = Array.from(rtByChoices.keys()).sort((a, b) => a - b);

  // Need at least 3 distinct choice levels to fit a sensible line
  if (uniqueChoiceLevels.length < 3) {
    return {
      ...FALLBACK_HICKS,
      stats: {
        totalTrials,
        excludedIncorrect,
        excludedOutOfRange,
        includedTrials: filteredTrials.length,
      },
    };
  }

  const xValues = uniqueChoiceLevels.map(n => Math.log2(n));
  const yValues = uniqueChoiceLevels.map(n => {
    const rts = rtByChoices.get(n)!;
    const sum = rts.reduce((acc, v) => acc + v, 0);
    return sum / rts.length;
  });

  const equation = linearRegression(xValues, yValues);
  return {
    ...equation,
    stats: {
      totalTrials,
      excludedIncorrect,
      excludedOutOfRange,
      includedTrials: filteredTrials.length,
    },
  };
}

// Predict movement time using Fitts' Law
export function predictFittsTime(
  equation: FittsLawEquation, 
  distance: number, 
  width: number
): number {
  const id = calculateIndexOfDifficulty(distance, width);
  return Math.max(0, equation.a + equation.b * id);
}

// Predict reaction time using Hick's Law
export function predictHicksTime(equation: HicksLawEquation, numChoices: number): number {
  return Math.max(0, equation.a + equation.b * Math.log2(numChoices));
}

// Get predicted time for a task based on its type
export function getPredictedTimeForTask(
  taskType: TaskType,
  fittsEquation?: FittsLawEquation,
  hicksEquation?: HicksLawEquation
): number | undefined {
  // Default task parameters for predictions
  const taskParams: Record<TaskType, { type: 'fitts' | 'hicks' | 'mixed'; params: any }> = {
    'button-click': {
      type: 'fitts',
      params: { distance: 200, width: 80 } // Typical button parameters
    },
    'drag-drop': {
      type: 'fitts',
      params: { distance: 250, width: 100 } // Drag distance and target size
    },
    'list-select': {
      type: 'mixed',
      params: {
        fitts: { distance: 150, width: 40 },
        choices: 4 // Number of list options
      }
    },
    'form-input': {
      type: 'mixed',
      params: {
        fitts: { distance: 100, width: 200 },
        choices: 2 // Input + submit
      }
    },
    "visual-search": {
      type: "fitts",
      params: undefined
    },
    "choice-reaction": {
      type: "fitts",
      params: undefined
    }
  };

  const config = taskParams[taskType];
  
  if (config.type === 'fitts' && fittsEquation) {
    return Math.round(predictFittsTime(fittsEquation, config.params.distance, config.params.width));
  } else if (config.type === 'hicks' && hicksEquation) {
    return Math.round(predictHicksTime(hicksEquation, config.params.choices));
  } else if (config.type === 'mixed' && fittsEquation && hicksEquation) {
    const fittsTime = predictFittsTime(fittsEquation, config.params.fitts.distance, config.params.fitts.width);
    const hicksTime = predictHicksTime(hicksEquation, config.params.choices);
    return Math.round(fittsTime + hicksTime);
  }
  
  return undefined;
}

// Generate Fitts' Law trial configurations
export function generateFittsTrials(): { width: number; distance: number }[] {
  const widths = [20, 40, 60, 80, 100]; // Target widths in pixels
  const distances = [100, 200, 300, 400, 500]; // Target distances in pixels
  const trials: { width: number; distance: number }[] = [];

  // Create trials for each combination, with repetitions
  for (const width of widths) {
    for (const distance of distances) {
      // 2 repetitions per combination
      trials.push({ width, distance });
      trials.push({ width, distance });
    }
  }

  // Shuffle trials
  return shuffleArray(trials);
}

// Generate Hick's Law trial configurations
export function generateHicksTrials(): {
  numChoices: number;
  targetKey: string;
  rangeStartIndex: number;
}[] {
  const allKeys = ["1", "2", "3", "4", "5", "6", "7"];
  const trials: {
    numChoices: number;
    targetKey: string;
    rangeStartIndex: number;
  }[] = [];
  // Use 5 distinct choice levels with 8 repetitions each = 40 trials total.
  const choiceLevels = [2, 3, 4, 5, 6];

  for (const numChoices of choiceLevels) {
    for (let rep = 0; rep < 8; rep++) {
      const rangeStartIndex = Math.floor(
        Math.random() * (allKeys.length - numChoices + 1)
      );

      const activeKeys = allKeys.slice(
        rangeStartIndex,
        rangeStartIndex + numChoices
      );

      const targetKey =
        activeKeys[Math.floor(Math.random() * activeKeys.length)];

      trials.push({
        numChoices,
        targetKey,
        rangeStartIndex,
      });
    }
  }

  return shuffleArray(trials);
}

function shuffleArray<T>(array: T[]): T[] {
  const shuffled = [...array];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  return shuffled;
}
