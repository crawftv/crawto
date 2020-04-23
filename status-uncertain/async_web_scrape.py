import asyncio
import requests
import nest_asyncio
from concurrent.futures import ThreadPoolExecutor

nest_asyncio.apply()


async def create_scrape_loop(
    iterable, individual_scrape_function, num_workers=40, *args
):
    """This function is responsible for organizing the asyncrhonous pieces.
    It creates a session, uses a list comprehension to create the tasks,
    and gathers the tasks.

    Parameters
    ----------

    iterable : list, default = None
        This a list of things to repeat with the individual_scrape_function.
        It can be, for example, a list of urls or page numbers.

    individual_scrape_function : function, default = None
        A function designed for scraping a single thing, usually a page or api request.
        Takes an element from the iterable and uses that to do multiple operations.



    num_workers : <class 'int'>, default = 40
        #TODO description

    Returns
    -------
    """
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        with requests.Session() as session:
            loop = asyncio.get_event_loop()
            tasks = [
                loop.run_in_executor(executor, individual_scrape_function, *(i, *args))
                for i in iterable
            ]
            for response in await asyncio.gather(*tasks):
                pass


def async_web_scrape(iterable, individual_scrape_function, num_workers, *args):
    """This function takes a function designed to scrape a single thing,
    i.e. a web page or api request. The individual_scrape_function gets
    passed to the the create_loop_function with the iterable and other args.

    Parameters
    ----------

    iterable : list, default = None
        This a list of things to repeat with the individual_scrape_function.
        It can be, for example, a list of urls or page numbers.

    individual_scrape_function : function, default = None
        A function designed for scraping a single thing, usually a page or api request.
        Takes an element from the iterable and uses that to do multiple operations.

    num_workers : <class 'int'>, default = 40
        The number of workers for the thread. This is passed directly to the 
        create_scrape_loop function.

    *args : these are the parameters of the individual scrape function.

    Returns
    -------
        Nothing. Unfortunately, this happens to use side effects. 
        Maybe in the future this can be changed.
    """
    future = asyncio.ensure_future(
        create_scrape_loop(iterable, individual_scrape_function, num_workers, *args)
    )
    loop = asyncio.get_event_loop()
    loop.run_until_complete(future)
