from abc import ABC, abstractmethod
import time
import os

class BaseLLMAPI(ABC):
    def __init__(
        self,
        model_pt: str,
        key_path: str,
        account_path: str | None = None,
        parallel_size: int = 1,
        max_retries: int = 10,
        initial_wait_time: int = 2,
        end_wait_time: int = 0,
    ):
        """
        Initializes the BaseLLMAPI object for calling API services.

        Args:
            model_pt (str): Model name
            key_path (str): Path to the key file (should be kept secret)
            account_path (str | None, optional): Path to the account file (should be kept secret). Defaults to None.
            parallel_size (int, optional): Number of parallel processes. Defaults to 1.
            max_retries (int, optional): Maximum number of retries. Defaults to 10.
            initial_wait_time (int, optional): Initial wait time. Defaults to 2.
            end_wait_time (int, optional): The wait time after the successful request. Defaults to 0.
        """

        self.model_name = model_pt

        if key_path :
            with open(key_path, "r") as f:
                self.key = f.read().strip()
        if account_path:
            with open(account_path, "r") as f:
                self.account = f.read().strip()

        self.parallel_size = parallel_size
        self.max_retries = max_retries
        self.initial_wait_time = initial_wait_time
        self.end_wait_time = end_wait_time

    @abstractmethod
    def _get_response(
        self,
        prompt: list[dict],
        n: int = 1,
        max_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: float = 1.0,
        logprobs: int | None = None,
    ) -> list[dict]:
        """
        Get the response from the API service.

        Args:
            prompt (list[dict]): The prompt to be sent to the API service.
            n (int, optional): Number of text generations per prompt. Defaults to 1.
            max_tokens (int, optional): Maximum number of tokens in the generated text. Defaults to 1024.
            temperature (float, optional): Controls the randomness of the generated text. Higher values make the text more random. Defaults to 1.0.
            top_p (float, optional): Controls the diversity of the generated text. Lower values make the text more focused. Defaults to 1.0.
            logprobs (int | None, optional): Number of log probabilities to include in the generated text. Defaults to None.

        Returns:
            list[dict]: The response from the API service. Each response is a dictionary containing the generated text ("text"), log probabilities ("logprobs", optional), and tokens ("tokens", optional).
        """
        pass

    def get_response(
        self,
        prompt: list[dict],
        n: int = 1,
        max_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: float = 1.0,
        logprobs: int | None = None,
    ) -> list[dict]:
        """
        Get the response from the API service.

        Args:
            prompt (list[dict]): The prompt to be sent to the API service.
            n (int, optional): Number of text generations per prompt. Defaults to 1.
            max_tokens (int, optional): Maximum number of tokens in the generated text. Defaults to 1024.
            temperature (float, optional): Controls the randomness of the generated text. Higher values make the text more random. Defaults to 1.0.
            top_p (float, optional): Controls the diversity of the generated text. Lower values make the text more focused. Defaults to 1.0.
            logprobs (int | None, optional): Number of log probabilities to include in the generated text. Defaults to None.

        Returns:
            list[dict]: The response from the API service. Each response is a dictionary containing the generated text ("text"), log probabilities ("logprobs", optional), and tokens ("tokens", optional).
        """
        retries = 0
        wait_time = self.initial_wait_time
        while retries < self.max_retries:
            try:
                response = self._get_response(
                    prompt=prompt,
                    n=n,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    logprobs=logprobs,
                )
                if self.end_wait_time > 0:
                    time.sleep(self.end_wait_time)
                return response
            except Exception as e:
                print(e)
                if retries == self.max_retries - 1:
                    # raise e
                    print("!!!WARNING: Max retries reached, returning empty response")
                    response = [{"text": ""}]
                    return response
                print("retrying...", retries, "sleep...", wait_time)
                time.sleep(wait_time)
                retries += 1
                wait_time *= 2


class GPTProxy(BaseLLMAPI):
    def __init__(
        self,
        model_pt: str,
        key: str,
        account: str,
        parallel_size: int,
        max_retries: int = 10,
        initial_wait_time: int = 2,
        end_wait_time: int = 0,
        port: int = 5050,
    ):
        """
        Initializes the BaseLLMAPI object for calling API services.

        Args:
            model_pt (str): Model name
            key_path (str): Path to the key file (should be kept secret)
            account_path (str): Path to the account file (should be kept secret)
            parallel_size (int): Number of parallel processes
            max_retries (int, optional): Maximum number of retries. Defaults to 10.
            initial_wait_time (int, optional): Initial wait time. Defaults to 2.
            end_wait_time (int, optional): End wait time. Defaults to 0.
            port (int, optional): Port number. Defaults to 5050.
        """
        from openai_client import OpenAIClient

        self.model_name = model_pt
        if os.path.exists(key):
            with open(key, "r") as f:
                api_key = f.read().strip()
        else:
            api_key = key
        if os.path.exists(account):
            with open(account, "r") as f:
                user = f.read().strip()
        else:
            user = account
        self.client = OpenAIClient(user=user, key=api_key, port=port)
        self.parallel_size = parallel_size
        self.max_retries = max_retries
        self.initial_wait_time = initial_wait_time
        self.end_wait_time = end_wait_time

    def _get_response(
        self,
        prompt: list[dict],
        n: int = 1,
        max_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: float = 1.0,
        logprobs: int | None = None,
    ) -> list[dict]:
        """
        Get the response from the API service.

        Args:
            prompt (list[dict]): The prompt to be sent to the API service.
            n (int, optional): Number of text generations per prompt. Defaults to 1.
            max_tokens (int, optional): Maximum number of tokens in the generated text. Defaults to 1024.
            temperature (float, optional): Controls the randomness of the generated text. Higher values make the text more random. Defaults to 1.0.
            top_p (float, optional): Controls the diversity of the generated text. Lower values make the text more focused. Defaults to 1.0.
            logprobs (int | None, optional): Number of log probabilities to include in the generated text. Defaults to None.

        Returns:
            list[dict]: The response from the API service. Each response is a dictionary containing the generated text ("text"), log probabilities ("logprobs", optional), and tokens ("tokens", optional).
        """
        if prompt[-1]["role"] != "user":
            raise ValueError("Last message should be user")
        response = self.client.chat_completions_create(
            model=self.model_name,
            messages=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=logprobs is not None,
            top_logprobs=logprobs,
            n=n,
            top_p=top_p,
        )
        choices = response["choices"]
        for i in range(len(choices)):
            choices[i]["text"] = choices[i]["message"]["content"]
            if "logprobs" in choices[i] and choices[i]["logprobs"] is not None:
                choices[i]["raw_logprobs"] = choices[i]["logprobs"]
                choices[i]["tokens"] = [
                    x["token"] for x in choices[i]["logprobs"]["content"]
                ]
                _logprobs = []
                for x in choices[i]["logprobs"]["content"]:
                    token_logprobs = {
                        y["token"]: y["logprob"] for y in x["top_logprobs"]
                    }
                    token_logprobs[x["token"]] = x["logprob"]
                    _logprobs.append(token_logprobs)
                choices[i]["logprobs"] = _logprobs
            else:
                choices[i]["logprobs"] = None
                choices[i]["tokens"] = None
        return choices
    

class GPT(BaseLLMAPI):
    def __init__(
        self,
        model_pt: str,
        key: str,
        account: str | None,
        parallel_size: int,
        max_retries: int = 10,
        initial_wait_time: int = 2,
        end_wait_time: int = 0,
        base_url: str | None = None,
    ):
        """
        Initializes the BaseLLMAPI object for calling API services.

        Args:
            model_pt (str): Model name
            key (str): Path to the key file (should be kept secret) or the API key
            account (str): Path to the account file (should be kept secret) or the organization ID
            parallel_size (int): Number of parallel processes
            max_retries (int, optional): Maximum number of retries. Defaults to 10.
            initial_wait_time (int, optional): Initial wait time. Defaults to 2.
            end_wait_time (int, optional): End wait time. Defaults to 0.
        """
        from openai import OpenAI

        self.model_name = model_pt
        if os.path.exists(key):
            with open(key, "r") as f:
                api_key = f.read().strip()
        else:
            api_key = key
        if account is None:
            organization = None
        elif os.path.exists(account):
            with open(account, "r") as f:
                organization = f.read().strip()
        else:
            organization = account
        if base_url is not None:
            self.client = OpenAI(api_key=api_key, organization=organization, base_url=base_url)
        else:
            self.client = OpenAI(api_key=api_key, organization=organization)
        self.parallel_size = parallel_size
        self.max_retries = max_retries
        self.initial_wait_time = initial_wait_time
        self.end_wait_time = end_wait_time

    def _get_response(
        self,
        prompt: list[dict],
        n: int = 1,
        max_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: float = 1.0,
        logprobs: int | None = None,
        response_format: str | None = None,
    ) -> list[dict]:
        """
        Get the response from the API service.

        Args:
            prompt (list[dict]): The prompt to be sent to the API service.
            n (int, optional): Number of text generations per prompt. Defaults to 1.
            max_tokens (int, optional): Maximum number of tokens in the generated text. Defaults to 1024.
            temperature (float, optional): Controls the randomness of the generated text. Higher values make the text more random. Defaults to 1.0.
            top_p (float, optional): Controls the diversity of the generated text. Lower values make the text more focused. Defaults to 1.0.
            logprobs (int | None, optional): Number of log probabilities to include in the generated text. Defaults to None.

        Returns:
            list[dict]: The response from the API service. Each response is a dictionary containing the generated text ("text"), log probabilities ("logprobs", optional), and tokens ("tokens", optional).
        """
        if prompt[-1]["role"] != "user":
            raise ValueError("Last message should be user")
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=logprobs is not None,
            top_logprobs=logprobs,
            n=n,
            top_p=top_p,
            response_format=response_format,
        )
        response = response.model_dump()
        choices = response["choices"]
        for i in range(len(choices)):
            choices[i]["text"] = choices[i]["message"]["content"]
            if "logprobs" in choices[i] and choices[i]["logprobs"] is not None:
                choices[i]["raw_logprobs"] = choices[i]["logprobs"]
                choices[i]["tokens"] = [
                    x["token"] for x in choices[i]["logprobs"]["content"]
                ]
                _logprobs = []
                for x in choices[i]["logprobs"]["content"]:
                    token_logprobs = {
                        y["token"]: y["logprob"] for y in x["top_logprobs"]
                    }
                    token_logprobs[x["token"]] = x["logprob"]
                    _logprobs.append(token_logprobs)
                choices[i]["logprobs"] = _logprobs
            else:
                choices[i]["logprobs"] = None
                choices[i]["tokens"] = None
        return choices
    