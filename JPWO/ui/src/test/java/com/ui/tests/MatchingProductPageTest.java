package com.ui.tests;

import com.ui.pages.ProductPage;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.chrome.ChromeDriver;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertTrue;

public class MatchingProductPageTest {

    private WebDriver driver;
    private ProductPage productPage;

    @BeforeEach
    void setUp() {
        driver = new ChromeDriver();
        driver.manage().window().maximize();
        driver.get("http://localhost:4200/");
        productPage = new ProductPage(driver);
    }

    @Test
    void shouldReturnMatchingProductsOnSearch() {
        productPage.search("screws");

        List<String> names = productPage.getVisibleProductNames();

        assertTrue(names.stream().allMatch(n -> n.toLowerCase().contains("screws")),
                "Search returned unrelated products");
    }

    @AfterEach
    void tearDown() {
        driver.quit();
    }
}
