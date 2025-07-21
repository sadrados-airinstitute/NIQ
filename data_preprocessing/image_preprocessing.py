import cv2
import pandas as pd
from utils.config import MAX_CONCURRENT_REQUESTS, MAX_RETRIES, RETRY_DELAY, TIMEOUT, IMAGE_SAVE_FOLDER
import os
import asyncio
import aiohttp
import nest_asyncio
import time
import os
import pandas as pd
from typing import List, Tuple, Iterator
from aiohttp import ClientSession


def preprocess_image(image):
    """
    Convert the input image to grayscale and upscale it to enhance OCR accuracy.

    Args:
        image (np.ndarray): Input color image in BGR format (as loaded by OpenCV).

    Returns:
        np.ndarray: Preprocessed grayscale and resized image ready for OCR.
    """
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return resized


def rotate_image(img, angle):
    """
    Rotate the image by a given angle (0, 90, 180, or 270 degrees).

    Args:
        img (np.ndarray): Input image (grayscale or color).
        angle (int): Rotation angle in degrees. Supported values: 0, 90, 180, 270.

    Returns:
        np.ndarray: Rotated image. If angle is 0 or unsupported, the original image is returned.
    """
    
    if angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        return img  # Return original if angle is 0


async def check_and_download_url(session: ClientSession, url: str, index: int, semaphore: asyncio.Semaphore) -> Tuple[int, bool, str]:
    """
    Attempts to download a single image from a URL with retry logic and concurrency control.

    Args:
        session (ClientSession): An active aiohttp client session for making HTTP requests.
        url (str): The image URL to download.
        index (int): Original index of the URL in the DataFrame (used for result alignment).
        semaphore (asyncio.Semaphore): Semaphore for limiting concurrency across requests.

    Returns:
        Tuple[int, bool, str]: A tuple containing:
            - index (int): The original row index.
            - success (bool): True if the image was successfully downloaded, False otherwise.
            - url (str): The image URL that was attempted.
    """
    attempt = 0
    while attempt < MAX_RETRIES:
        try:
            async with semaphore:
                async with session.get(url, timeout=TIMEOUT) as response:
                    if response.status == 200:
                        content = await response.read()
                        os.makedirs(IMAGE_SAVE_FOLDER, exist_ok=True)
                        image_path = os.path.join(IMAGE_SAVE_FOLDER, f"image_{index}.jpg")
                        with open(image_path, 'wb') as f:
                            f.write(content)
                        print(f"[✓] Downloaded: {url}")
                        return index, True, url
                    else:
                        print(f"[x] HTTP {response.status} from {url} (attempt {attempt + 1})")
        except asyncio.TimeoutError:
            print(f"[!] Timeout while accessing {url} (attempt {attempt + 1})")
        except Exception as e:
            print(f"[!] Error accessing {url} (attempt {attempt + 1}): {e}")
        attempt += 1
        await asyncio.sleep(RETRY_DELAY)

    print(f"[x] Failed after {MAX_RETRIES} attempts: {url}")
    return index, False, url


async def check_and_download_urls_batch(session: ClientSession, batch: List[Tuple[int, str]], semaphore: asyncio.Semaphore, processed: List[int]) -> List[Tuple[int, bool, str]]:
    """
    Attempts to download a single image from a URL with retry logic and concurrency control.

    Args:
        session (ClientSession): An active aiohttp client session for making HTTP requests.
        url (str): The image URL to download.
        index (int): Original index of the URL in the DataFrame (used for result alignment).
        semaphore (asyncio.Semaphore): Semaphore for limiting concurrency across requests.

    Returns:
        Tuple[int, bool, str]: A tuple containing:
            - index (int): The original row index.
            - success (bool): True if the image was successfully downloaded, False otherwise.
            - url (str): The image URL that was attempted.
    """
    results = await asyncio.gather(
        *(check_and_download_url(session, url, index, semaphore) for index, url in batch)
    )
    processed.extend([idx for idx, _ in batch])
    print(f"[•] Processed batch: {batch[0][0]} to {batch[-1][0]}")
    return results


def chunk_urls_with_index(df: pd.DataFrame, batch_size: int = 100) -> Iterator[List[Tuple[int, str]]]:
    """
    Splits the image URL column into batches with index tracking.

    Args:
        df (pd.DataFrame): Input DataFrame containing 'image_nutrition_url' column.
        batch_size (int, optional): Number of URLs per batch. Defaults to 100.

    Yields:
        List[Tuple[int, str]]: A list of (index, url) pairs for each batch.
    """
    for i in range(0, len(df), batch_size):
        yield list(zip(df.index[i:i + batch_size], df['image_nutrition_url'].iloc[i:i + batch_size]))


async def main_async(df: pd.DataFrame) -> pd.DataFrame:
    """
    Asynchronously downloads all images from the 'image_nutrition_url' column in the input DataFrame.

    Uses batched concurrent downloads with retries and timeout handling.
    Updates the DataFrame with an 'accessibility' column indicating success/failure.

    Args:
        df (pd.DataFrame): Input DataFrame with a required 'image_nutrition_url' column.
                           Assumes this column contains HTTP(S) links to images.

    Returns:
        pd.DataFrame: A filtered DataFrame including only accessible/downloaded image rows.
                      Adds 'accessibility' column with True for success, False for failure.
    """
    start_time = time.time()
    processed_urls = []

    async with aiohttp.ClientSession() as session:
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

        batch_tasks = [
            check_and_download_urls_batch(session, batch, semaphore, processed_urls)
            for batch in chunk_urls_with_index(df)
        ]
        batch_results = await asyncio.gather(*batch_tasks)

    all_results = [result for batch in batch_results for result in batch]
    all_results_sorted = sorted(all_results, key=lambda x: x[0])

    accessibility_flags = [res[1] for res in all_results_sorted]
    failed_urls = [res[2] for res in all_results_sorted if not res[1]]

    df['accessibility'] = pd.Series(accessibility_flags, index=df.index)
    df_success = df[df['accessibility']].copy()

    print(f"[✓] Downloaded: {len(df_success)} images")
    print(f"[x] Failed: {len(failed_urls)} images")
    if failed_urls:
        print(f"[!] Failed URLs (showing first 5): {failed_urls[:5]}")

    print(f"[✓] Total time: {time.time() - start_time:.2f} seconds")
    return df_success


def download_images_from_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Orchestrates the image downloading pipeline for a DataFrame with image URLs.

    This function asynchronously downloads images from the URLs specified in the
    'image_nutrition_url' column of the input DataFrame. It filters out any rows
    where the download fails (due to network errors, timeouts, etc.), and returns
    only the successful entries.

    Args:
        df (pd.DataFrame): Input DataFrame. Must contain a column named 'image_nutrition_url'
                           with valid HTTP/HTTPS image URLs.

    Returns:
        pd.DataFrame: A filtered DataFrame including only the rows where the image
                      was successfully downloaded and marked as accessible. Includes
                      an added 'accessibility' boolean column.

    Raises:
        ValueError: If 'image_nutrition_url' column is not present in the DataFrame.
    """
    if 'image_nutrition_url' not in df.columns:
        raise ValueError("DataFrame must contain a column named 'image_nutrition_url'.")
    return asyncio.run(main_async(df))