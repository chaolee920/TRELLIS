import os
import pybase64
import requests
from time import time
import asyncio
import json

# Meshy AI API configuration
MESHY_API_KEY = os.getenv('MESHY_API_KEY')  # Set your Meshy API key as environment variable
MESHY_API_BASE = 'https://api.meshy.ai/v2/text-to-3d'
MESHY_HEADERS = {
    'Authorization': f'Bearer {MESHY_API_KEY}',
    'Content-Type': 'application/json'
}

if not MESHY_API_KEY:
    raise ValueError("Please set MESHY_API_KEY environment variable")


def create_meshy_task(prompt):
    """Create a text-to-3D task using Meshy AI API"""
    payload = {
        'mode': 'preview',  # or 'refine' for higher quality
        'prompt': prompt + ", 3d style, whole body, cartoon asset",
        'art_style': 'cartoon',
        'negative_prompt': 'low quality, blurry, distorted'
    }
    
    response = requests.post(MESHY_API_BASE, headers=MESHY_HEADERS, json=payload)
    
    if response.status_code != 200:
        raise Exception(f"Failed to create Meshy task: {response.status_code} - {response.text}")
    
    return response.json()


def get_meshy_task_status(task_id):
    """Get the status of a Meshy AI task"""
    url = f"{MESHY_API_BASE}/{task_id}"
    response = requests.get(url, headers=MESHY_HEADERS)
    
    if response.status_code != 200:
        raise Exception(f"Failed to get Meshy task status: {response.status_code} - {response.text}")
    
    return response.json()


def download_meshy_result(download_url, filename):
    """Download the 3D model from Meshy AI"""
    response = requests.get(download_url, stream=True)
    
    if response.status_code != 200:
        raise Exception(f"Failed to download Meshy result: {response.status_code}")
    
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    return filename


async def wait_for_meshy_completion(task_id, max_wait_time=600, poll_interval=10):
    """Wait for Meshy AI task to complete with polling"""
    start_time = time()
    
    while time() - start_time < max_wait_time:
        task_status = get_meshy_task_status(task_id)
        status = task_status.get('status')
        
        print(f"Meshy task {task_id} status: {status}")
        
        if status == 'SUCCEEDED':
            return task_status
        elif status == 'FAILED':
            raise Exception(f"Meshy task failed: {task_status.get('error', 'Unknown error')}")
        
        await asyncio.sleep(poll_interval)
    
    raise Exception(f"Meshy task timed out after {max_wait_time} seconds")


def validate(prompt, result_path="./sample.ply"):
    with open(result_path, "rb") as file:
        file_data = file.read()
    encoded_data = pybase64.b64encode(file_data).decode("utf-8")
    validate_url = 'http://127.0.0.1:8094/validate_txt_to_3d_ply'
    response = requests.post(validate_url, json={"prompt": prompt, "data": encoded_data})
    if response.status_code == 200:
        results_validation = response.json()

        validation_score = float(results_validation["score"])
        return validation_score
    else:
        print("Validation failed with status code:", response.status_code)
        return 0


async def _generate_t23_sync(prompt):
    """Generate 3D model using Meshy AI text-to-3D API"""
    try:
        # Create the task
        print(f"Creating Meshy AI task for prompt: {prompt}")
        task_response = create_meshy_task(prompt)
        task_id = task_response['result']
        
        print(f"Meshy task created with ID: {task_id}")
        
        # Wait for completion
        completed_task = await wait_for_meshy_completion(task_id)
        
        # Download the result
        model_urls = completed_task.get('model_urls', {})
        ply_url = model_urls.get('ply')
        
        if not ply_url:
            print("No PLY URL found in response")
            return None
        
        # Download the PLY file
        filename = "sample_t23.ply"
        download_meshy_result(ply_url, filename)
        
        # Validate the result
        score = validate(prompt, filename)
        print(f"Score from text-to-3d: {score}")
        
        return (filename, score)
        
    except Exception as e:
        print(f"Error during Meshy generation: {e}")
        return None

async def generate_t23(prompt):
    return await _generate_t23_sync(prompt)


async def _generate_t2i23_sync(prompt, guidance_scale=7.5, num_inference_steps=25):
    """Generate 3D model using Meshy AI with refined settings"""
    try:
        # Create a task with refined mode for better quality
        print(f"Creating refined Meshy AI task for prompt: {prompt}")
        payload = {
            'mode': 'refine',  # Higher quality mode
            'prompt': prompt + ", white background, 3d style, whole body, cartoon asset, detailed",
            'art_style': 'realistic',  # Different style for variety
            'negative_prompt': 'low quality, blurry, distorted, incomplete, broken'
        }
        
        response = requests.post(MESHY_API_BASE, headers=MESHY_HEADERS, json=payload)
        
        if response.status_code != 200:
            raise Exception(f"Failed to create refined Meshy task: {response.status_code} - {response.text}")
        
        task_response = response.json()
        task_id = task_response['result']
        
        print(f"Refined Meshy task created with ID: {task_id}")
        
        # Wait for completion (refined mode takes longer)
        completed_task = await wait_for_meshy_completion(task_id, max_wait_time=900)
        
        # Download the result
        model_urls = completed_task.get('model_urls', {})
        ply_url = model_urls.get('ply')
        
        if not ply_url:
            print("No PLY URL found in refined response")
            return None
        
        # Download the PLY file
        filename = "sample_t2i23.ply"
        download_meshy_result(ply_url, filename)
        
        # Validate the result
        score = validate(prompt, filename)
        print(f"Score from refined text-to-3d: {score}")
        
        return (filename, score)
        
    except Exception as e:
        print(f"Error during refined Meshy generation: {e}")
        return None

async def generate_t2i23(prompt, guidance_scale=7.5, num_inference_steps=25):
    return await _generate_t2i23_sync(prompt, guidance_scale, num_inference_steps)


async def main():
    prompt = "pink bicycle"
    print("=============================================================")
    print("Using Meshy AI external API for text-to-3D generation")

    print(f"====Prompt: {prompt}====")
    t0 = time()

    # Run both generation methods concurrently with early termination
    t23_task = asyncio.create_task(generate_t23(prompt))
    t2i23_task = asyncio.create_task(generate_t2i23(prompt, guidance_scale=9.0, num_inference_steps=20))
    
    pending_tasks = {t23_task, t2i23_task}
    task_names = {t23_task: 't23', t2i23_task: 't2i23'}
    
    output_t23, score_t23 = None, 0
    output_t2i23, score_t2i23 = None, 0
    best_gaussian, best_score = None, 0
    early_termination = False
    
    # Process results as they complete
    while pending_tasks:
        done, pending_tasks = await asyncio.wait(pending_tasks, return_when=asyncio.FIRST_COMPLETED)
        
        for completed_task in done:
            try:
                result = await completed_task
                task_type = task_names[completed_task]
                
                if result is not None:
                    filename, score = result
                    
                    if task_type == 't23':
                        output_t23, score_t23 = filename, score
                        print(f"generate_t23 completed with score: {score}")
                        
                        # Early termination if t23 score > 0.65
                        if score > 0.65:
                            print(f"Early termination: t23 score {score} > 0.65, cancelling t2i23")
                            best_gaussian, best_score = filename, score
                            early_termination = True
                            
                            # Cancel remaining tasks
                            for task in pending_tasks:
                                task.cancel()
                            pending_tasks.clear()
                            break
                        
                    elif task_type == 't2i23':
                        output_t2i23, score_t2i23 = filename, score
                        print(f"generate_t2i23 completed with score: {score}")
                else:
                    print(f"Generation method {task_type} returned None")
                    
            except Exception as e:
                task_type = task_names[completed_task]
                print(f"Error in {task_type}: {e}")
    
    # If not early terminated, select the best result
    if not early_termination:
        if output_t23 is None and output_t2i23 is None:
            raise Exception("Both generation methods failed")
        elif output_t23 is None:
            best_gaussian = output_t2i23
            best_score = score_t2i23
        elif output_t2i23 is None:
            best_gaussian = output_t23
            best_score = score_t23
        elif score_t23 < score_t2i23:
            best_gaussian = output_t2i23
            best_score = score_t2i23
        else:
            best_gaussian = output_t23
            best_score = score_t23
    
    t1 = time()
    termination_status = "early termination" if early_termination else "both methods completed"
    print(f"====Final Score: {best_score}, Generation took: {t1 - t0:.2f}s, Status: {termination_status}====")
    print(f"Best result saved as: {best_gaussian}")
    print("=============================================================")
    return best_gaussian, best_score

if __name__ == "__main__":
    asyncio.run(main())
