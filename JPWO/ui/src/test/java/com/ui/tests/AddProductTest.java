package com.ui.tests;

import com.ui.pages.AddProductPage;
import com.ui.pages.LoginPage;
import com.ui.pages.ProductAdminPage;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.chrome.ChromeDriver;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class AddProductTest {

    private WebDriver driver;
    private LoginPage loginPage;
    private AddProductPage addProductPage;
    private ProductAdminPage productAdminPage;

    @BeforeEach
    void setUp() {
        driver = new ChromeDriver();
        driver.manage().window().maximize();
        driver.get("http://localhost:4200/auth/login");
        loginPage = new LoginPage(driver);
        loginPage.loginAsAdmin("admin@practicesoftwaretesting.com", "welcome01");
        addProductPage = new AddProductPage(driver);
        productAdminPage = new ProductAdminPage(driver);
    }

    @Test
    void shouldAddNewProduct() throws InterruptedException {
        // SPRÓBOWAĆ TUTAJ JAKIEJŚ UNIKALNEJ NAZWY ZEBY NIE BYLO ZE STALA NAZWA JUZ ISTNIEJE
        addProductPage.open();
        addProductPage.fillForm("Nowy Produkt", "Opis nowego produktu", 100, 29.99,
                "ForgeFlex Tools", "Hand Tools", "Hammer");
        productAdminPage.open();
        productAdminPage.search("Nowy Produkt");
        String addedProductName = productAdminPage.getProductName("Nowy Produkt");
        String addedProductPrice = productAdminPage.getProductPrice("Nowy Produkt");

        assertEquals("Nowy Produkt", addedProductName);
        assertEquals("$29.99", addedProductPrice);
    }

    @AfterEach
    void tearDown() {
        driver.quit();
    }
}
