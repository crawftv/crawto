import asyncio
from concurrent.futures import ThreadPoolExecutor
import nest_asyncio
def async_web_scrape(iterable, individual_scrape_function, *args):
    """
    This function does something analogous to compiling the get_data_asynchronously function,
    Then it executes loop.
    1. Call the get_data_function
    2. Get the event_loop
    3. Run the tasks (Much easier to understand in python 3.7, "ensure_future" was changed to "create_task")
    4. Edge_list and top_interactions will be passed to the next functions
    """
    nest_asyncio.apply()
    async def create_scrape_loop(iterable, individual_scrape_function, *args):
        """
        1. Establish an executor and number of workers
        2. Establish the session
        3. Establish the event loop
        4. Create the task by list comprenhensions
        5. Gather tasks.
        """
        with ThreadPoolExecutor(max_workers=40) as executor:
            with requests.Session() as session:
                loop = asyncio.get_event_loop()
                tasks = [
                    loop.run_in_executor(
                        executor, individual_scrape_function, *(i, *args)
                    )
                    for i in iterable
                ]
                for response in await asyncio.gather(*tasks):
                    pass
    
    future = asyncio.ensure_future(
        create_scrape_loop(iterable, individual_scrape_function,*args)
    )
    loop = asyncio.get_event_loop()
    loop.run_until_complete(future)